#! /usr/bin/python 
# -*- encoding: utf-8 -*-
'''
@author LeslieZhao
@date 20230620
'''
import torch
from trainer.ModelTrainer import ModelTrainer
from diffusers import StableDiffusionPipeline,DPMSolverMultistepScheduler
from model.network import Lora
from diffusers.optimization import get_scheduler
from utils.utils import *
from model.loss import *
import torch.nn.functional as F
from itertools import chain
import pdb

class DiffuserTrainer(ModelTrainer,Lora):

    def __init__(self, args):
        super().__init__(args)
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        

        self.pipeline = StableDiffusionPipeline.from_pretrained(args.basemodel,safety_checker=None).to(self.device)
        self.noise_scheduler = DPMSolverMultistepScheduler.from_config(
                                self.pipeline.scheduler.config, use_karras_sigmas=True
                            )
        self.text_encoder = self.pipeline.text_encoder.to(self.device)
        self.vae = self.pipeline.vae.to(self.device)
        self.unet = self.pipeline.unet.to(self.device)

        requires_grad(self.vae,False)
        requires_grad(self.text_encoder,False)
        
        self.unet_lora_layers = self.get_unet_lora_layer()
        
        if self.args.train_text_encoder:
            self.text_encoder_lora = self.get_text_encoder_lora_layer()

        self.optimG,self.scheduler_G = self.create_optimizer()
        self.unet,self.unet_modeule = self.use_ddp(self.unet,dist=args.dist)
        if self.args.train_text_encoder:
            requires_grad(self.text_encoder,True)
            self.text_encoder,self.text_encoder_modeule = self.use_ddp(self.text_encoder,dist=args.dist)
            self.pipeline.text_encoder = self.text_encoder_modeule

        if self.args.pretrain_path is not None:
            self.loadParameters(self.args.pretrain_path)

        
        self.pipeline.unet = self.unet_modeule
        self.pipeline = self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)
        self.pipeline.safety_checker = lambda images, clip_input: (images, None)

        if self.args.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()


    
        
    
    def create_optimizer(self):

        if self.args.train_text_encoder:
            params = chain(self.unet_lora_layers.parameters(),self.text_encoder_lora.parameters())
        else:
            params = self.unet_lora_layers.parameters()
        optimizer = torch.optim.AdamW(
                params,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                weight_decay=self.args.adam_weight_decay,
                eps=self.args.adam_epsilon,
            )
        
        lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps * self.args.gradient_accumulation_steps,
        )

        return optimizer,lr_scheduler

        
    def run_single_step(self, data, steps):
        
        data = self.process_input(data)
        self.run_generator_one_step(data,steps)
        

    def run_generator_one_step(self, data,step):
        
        self.unet.train()
        
        input_ids,pixel_values = data
      
        self.optimG.zero_grad()
        if self.args.mixed_precision:
            with torch.autocast(device_type=self.device, dtype=torch.float16):
                G_losses,loss = \
                    self.compute_g_loss(input_ids,pixel_values,step)
                
                self.scaler.scale(loss).backward()

                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimG)

                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(chain(self.unet_lora_layers.parameters(),self.text_encoder_lora.parameters()) if self.args.train_text_encoder \
                                               else self.unet_lora_layers.parameters()
                                               , self.args.max_grad_norm)

                # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                self.scaler.step(self.optimG)

                # Updates the scale for next iteration.
                self.scaler.update()
        else:
            G_losses,loss = \
                    self.compute_g_loss(input_ids,pixel_values,step)
            
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(chain(self.unet_lora_layers.parameters(),self.text_encoder_lora.parameters()) if self.args.train_text_encoder \
                                               else self.unet_lora_layers.parameters()
                                               , self.args.max_grad_norm)
            self.optimG.step()

        self.g_losses = reduce_loss_dict(G_losses)
        # print('after')
        # print('lora :',self.lora_layers.state_dict()['down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor.to_q_lora.down.weight'][:2,0])
        # print('pipeline:',list(self.unet_modeule.attn_processors['down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor'].state_dict().values())[0][:2,0])
        # print('unet :',list(self.pipeline.unet.attn_processors['down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor'].state_dict().values())[0][:2,0])
        self.generator = None
        
    
    def compute_g_loss(self,input_ids,pixel_values,step):
        
        G_losses = {}
        loss = 0

        model_pred,timesteps,target = self.forward(input_ids,pixel_values)
        if self.args.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(timesteps,self.noise_scheduler)
            mse_loss_weights = (
                torch.stack([snr, self.args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        G_losses['loss'] = loss
        return G_losses,loss

    def forward(self,input_ids,pixel_values):
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if self.args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.args.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )

        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        input_ids = input_ids.int()
        encoder_hidden_states = self.text_encoder(input_ids)[0]

        # Get the target for loss depending on the prediction type
        if self.args.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.noise_scheduler.register_to_config(prediction_type=self.args.prediction_type)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # Predict the noise residual and compute loss
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        return model_pred,timesteps,target

    def evalution(self,test_loader,steps,epoch):
        
        generator = torch.Generator(device=self.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        
        with torch.no_grad():
            images = []
            for _ in range(self.args.num_validation_images):
                images.append(
                    self.pipeline(self.args.test_caption, num_inference_steps=30, generator=generator,output_type='pt').images
                )
        if self.args.rank == 0:
            self.val_vis.display_current_results(self.select_img(images,name=self.args.test_caption),steps)

    def get_latest_losses(self):
        if not hasattr(self,'d_losses'):
            return self.g_losses
        return {**self.g_losses,**self.d_losses}

    def get_latest_generated(self):
        return self.generator

    def loadParameters(self,path):
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        self.unet_lora_layers.load_state_dict(ckpt['unet_lora'])
        
        self.optimG.load_state_dict(ckpt['g_optim'])
        self.scheduler_G.load_state_dict(ckpt['lr_scheduler'])

        if self.args.train_text_encoder: 
            self.text_encoder_lora.load_state_dict(ckpt['text_encoder_lora'])

    def saveParameters(self,path):
        save_dict = {
            'unet_lora':self.unet_lora_layers.state_dict(),
            "g_optim": self.optimG.state_dict(),
            'lr_scheduler':self.scheduler_G.state_dict(),
            "args": self.args
        }
        lora  = {'unet.%s'%(k):v for k,v in self.unet_lora_layers.state_dict().items()}
        if self.args.train_text_encoder:
            save_dict['text_encoder_lora'] = self.text_encoder_lora.state_dict()
            lora = {**lora,**{
                'text_encoder.%s'%(k):v for k,v in self.text_encoder_lora.state_dict().items()}
            }
        save_dict['lora'] = lora
        
        torch.save(
                   save_dict,
                   path
                )

    def get_lr(self):
        return self.optimG.state_dict()['param_groups'][0]['lr']





        


    
    

    
