'''
@author LeslieZhao
@date 20230620
'''

import torch
import pdb 
from diffusers import (StableDiffusionPipeline, 
                       DPMSolverMultistepScheduler,
                       ControlNetModel,
                       StableDiffusionControlNetPipeline,
                       StableDiffusionInpaintPipeline)

from model.custom import StableDiffusionInpaitAdapterPipeline,T2IAdapter
import argparse
import os
import numpy as np
import cv2
import PIL.Image as Image
from model.third.openpose import OpenposeDetector
from model.third.parsing.parsing import FaceParsing
from diffusers.utils import load_image
parser = argparse.ArgumentParser(description="infer")

parser.add_argument('--basemodel',default='pretrained_models/chilloutmixNiPruned_Tw1O',type=str,help='base model path')
parser.add_argument('--lora_path',default=None,type=str,help='lora model path')
parser.add_argument('--control_path',default=None,type=str,help='controlnet model path')
parser.add_argument('--inpait_path',default='pretrained_models/stable-diffusion-inpainting',type=str,help='inpait model path')
parser.add_argument('--adapter_ckpt',default="pretrained_models/t2iadapter_seg_sd14v1.pth",type=str,help='adapter model path')

parser.add_argument('--ref_img',default=None,type=str,help='ref image path')
parser.add_argument('--pose_img',default=None,type=str,help='pose image path')
parser.add_argument('--mask',default=None,type=str,help='mask image path')
parser.add_argument('--adapter_mask',default=None,type=str,help='adapter_mask path')
parser.add_argument('--mask_area',default='all',choices=['hair', 'bg','all'])
parser.add_argument('--mode',default='lora',choices=['lora', 'control','inpait','t2iinpait'])
parser.add_argument('--prompt',default=None,type=str,help='prompt')
parser.add_argument('--neg_prompt',default='(painting by bad-artist-anime:0.9), (painting by bad-artist:0.9), watermark, text, error, blurry, jpeg artifacts, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, artist name, (worst quality, low quality:1.4), bad anatomy, watermark, signature, text, logo',type=str,help='negative prompt')
parser.add_argument('--width',default=512,type=int,help='input image width')
parser.add_argument('--height',default=512,type=int,help='input image height')

parser.add_argument('--num_inference_steps',default=50,type=int,help='inference steps number')
parser.add_argument('--num_images_per_prompt',default=1,type=int,help='generate image number')
parser.add_argument('--seed',default=3728865715,type=int,help='random seed')
parser.add_argument('--guidance_scale',default=8,type=int,help='control picture quality')
parser.add_argument('--scale',default=1.2,type=float,help='mixed scale')
parser.add_argument('--outpath',default='./1.png',type=str,help='path to save image')

class Infer:
    def __init__(self,args):
        self.args = args
    def set_safety_checker(self):
        self.pipeline.safety_checker = lambda images, clip_input: (images, None)
    
    def get_input_kwargs(self):
        input_kwargs = {
            'prompt':self.args.prompt, 
            'negative_prompt':self.args.neg_prompt, 
            'width':self.args.width, 
            'height':self.args.height, 
            'num_inference_steps':self.args.num_inference_steps, 
            'num_images_per_prompt':self.args.num_images_per_prompt,
            'generator':torch.manual_seed(self.args.seed),
            'guidance_scale':self.args.guidance_scale,
            'cross_attention_kwargs':{"scale": self.args.scale},
            'output_type':'np'
        }
        return input_kwargs
    
    def run(self,input_kwargs):
        with torch.no_grad():
            images = self.pipeline(**input_kwargs).images
        images = np.concatenate(images,1)
        images = np.clip(images*255.,0,255).astype(np.uint8)
        images = cv2.cvtColor(images,cv2.COLOR_RGB2BGR)
        save_base = os.path.split(self.args.outpath)[0]
        os.makedirs(save_base,exist_ok=True)
        cv2.imwrite(self.args.outpath,images)

class LoraInfer(Infer):
    def __init__(self, args):
        super().__init__(args)
        self.pipeline = StableDiffusionPipeline.from_pretrained(args.basemodel,safety_checker=None).to('cuda')
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config, use_karras_sigmas=True
        )

        lora_state = torch.load(args.lora_path)['lora']
        self.pipeline.load_lora_weights(lora_state)
        self.set_safety_checker()

    def __call__(self):
        input_kwargs = self.get_input_kwargs()
        self.run(input_kwargs)


class ControlInfer(Infer,OpenposeDetector):
    def __init__(self, args):
        
        Infer.__init__(self,args)
        OpenposeDetector.__init__(self)
        controlnet = ControlNetModel.from_pretrained(
            args.control_path,
            torch_dtype=torch.float32,
            local_files_only=True,
        ).to('cuda')

        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            args.basemodel,
            controlnet=controlnet, torch_dtype=torch.float32,
            local_files_only=True,
        ).to('cuda')
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config, use_karras_sigmas=True
            )
        lora_state = torch.load(args.lora_path)['lora']
        self.pipeline.load_lora_weights(lora_state)
        self.set_safety_checker()

    def __call__(self):
        if self.args.pose_img is not None and os.path.exists(self.args.pose_img):
            pose = load_image(self.args.pose_img)
        else:
            ref_img = cv2.imread(self.args.ref_img)
            pose,_ = self.get_pose(ref_img,hand=True)
            pose = Image.fromarray(pose,mode='RGB')
        input_kwargs = self.get_input_kwargs()
        input_kwargs['image'] = pose
        
        self.run(input_kwargs)

class InpaitInfer(Infer,FaceParsing):
    def __init__(self, args):
        Infer.__init__(self,args)
        FaceParsing.__init__(self)

        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            self.args.inpait_path).to('cuda')
        self.set_safety_checker()
        self.index = []
        if self.args.mask_area == 'hair':
            self.index = [17]
        elif self.args.mask_area == 'bg':
            self.index = [0]
        else:
            self.index = [0,17]

    def __call__(self):
        img = cv2.imread(args.ref_img)
        img = cv2.resize(img,(512,512))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        inp = Image.fromarray(img,mode='RGB')
        
        if args.mask is None:
            h,w,_ = img.shape
            p_img = self.preprocess_parsing(img)
            parsing = self.get_parsing(p_img)
            parsing = self.postprocess_parsing(parsing,h,w)
            mask = np.zeros((h,w,3),dtype=np.uint8)
           
            for i in self.index:
                mask[parsing==i] = 255
            mask = mask.astype(np.uint8)
          
        else:
            mask = cv2.imread(args.mask)
        
        mask = Image.fromarray(mask,mode='RGB')
        input_kwargs = self.get_input_kwargs()
        input_kwargs['image'] = inp
        input_kwargs['mask_image'] = mask
        del input_kwargs['cross_attention_kwargs']
        self.run(input_kwargs)

class T2IInpaitInfer(Infer):
    def __init__(self, args):
        super().__init__(args)

        self.pipeline = StableDiffusionInpaitAdapterPipeline.from_pretrained(self.args.inpait_path).to('cuda')


        self.pipeline.adapter = T2IAdapter(
            block_out_channels=[320, 640, 1280, 1280][:4],
            channels_in=3, 
            num_res_blocks=2, 
            kernel_size=1, 
            res_block_skip=True, 
            use_conv=False
        )
        weight = self.convert_model()
        
        self.pipeline.adapter.load_state_dict(weight)
        self.pipeline.adapter = self.pipeline.adapter.to('cuda')
        self.set_safety_checker()

    def convert_model(self):
        weight = torch.load(args.adapter_ckpt)
        mapping = {}
        for k in weight.keys():

            if 'in_conv' in k:
                mapping[k] = k.replace('in_conv','conv1')
        for old, new in mapping.items():
            weight[new] = weight.pop(old)
        return weight
    def __call__(self):

        img = load_image(args.ref_img)
        mask = load_image(args.mask)
        adapter_mask = load_image(args.adapter_mask)
        
        input_kwargs = self.get_input_kwargs()
        input_kwargs['image'] = img
        input_kwargs['mask_image'] = mask
        input_kwargs['adapter_image'] = adapter_mask
        del input_kwargs['cross_attention_kwargs']
        self.run(input_kwargs)


if __name__ == "__main__":
    
    args = parser.parse_args()
    if args.mode == 'lora':
        infer = LoraInfer(args)
    elif args.mode == 'control':
        infer = ControlInfer(args)
    elif args.mode == 'inpait':
        infer = InpaitInfer(args)
    elif args.mode == 't2iinpait':
        infer = T2IInpaitInfer(args)
    infer()