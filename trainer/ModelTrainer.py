#! /usr/bin/python 
# -*- encoding: utf-8 -*-
'''
@author LeslieZhao
@date 20230529
'''
import torch 
import math
import time,os

from utils.visualizer import Visualizer
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.distributed as dist 
import subprocess
from utils.utils import convert_img
class ModelTrainer:

    def __init__(self,args):
        
        self.args = args
        self.batch_size = args.batch_size
        self.old_lr = args.lr
        if args.rank == 0 :
            self.vis = Visualizer(args)

            if args.eval:
                self.val_vis = Visualizer(args,"val")

    # ## ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== =====

    def train_network(self,train_loader,test_loader):

        counter = 0
        loss_dict = {}
        acc_num = 0
        mn_loss = float('inf')

        steps = 0
        begin_it = 0
        if self.args.pretrain_path:
            try:
                begin_it = int(self.args.pretrain_path.split('/')[-1].split('-')[0]) + 1
                steps = int(self.args.pretrain_path.split('/')[-1].split('-')[1].replace('.pth','')) + 1
                print("current steps: %d | one epoch steps: %d "%(steps,self.args.mx_data_length))
            except:
                steps = 0
                begin_it = 0
        if not self.args.apply_begin_it:
            begin_it = 0
            steps = 0
        for epoch in range(begin_it,self.args.max_epoch):

            for ii,(data) in enumerate(train_loader):
                
                tstart = time.time()
                
                self.run_single_step(data,steps)
                losses = self.get_latest_losses()

                for key,val in losses.items():
                    loss_dict[key] = loss_dict.get(key,0) + val.mean().item()

                counter += 1
                

                telapsed = time.time() - tstart


                if steps % self.args.print_interval == 0 and self.args.rank == 0:
                    
                    for key,val in loss_dict.items():
                        loss_dict[key] /= counter
                    lr_rate = self.get_lr()
                    print_dict = {**{"time":telapsed,"lr":lr_rate},
                                    **loss_dict}
                    self.vis.print_current_errors(epoch,steps,print_dict,telapsed)
                    
                    self.vis.plot_current_errors(print_dict,steps)

                    display_data = self.select_img(self.get_latest_generated())

                    self.vis.display_current_results(display_data,steps)
                    
                    loss_dict = {}
                    counter = 0
                    
                    # torch.cuda.empty_cache()
                if self.args.save_interval != 0 and steps % self.args.save_interval == 0 and \
                    self.args.rank == 0:
                    self.saveParameters(os.path.join(self.args.checkpoint_path,"%03d-%08d.pth"%(epoch,steps)))
                    self.current_epoch,self.current_step = epoch,steps
                    
            


                if self.args.eval and self.args.test_interval > 0 and steps % self.args.test_interval == 0:
                    val_loss = self.evalution(test_loader,steps,epoch)

                    if self.args.early_stop:
                        
                        acc_num,mn_loss,stop_flag = self.early_stop_wait(self.get_loss_from_val(val_loss),acc_num,mn_loss)
                        if stop_flag:
                            return 
       
                # print('******************memory:',psutil.virtual_memory()[3])
                if self.args.max_train_steps > 0 and steps > self.args.max_train_steps:
                    return 
                steps += 1
            if self.args.rank == 0 :
                self.saveParameters(os.path.join(self.args.checkpoint_path,"%03d-%08d.pth"%(epoch+1,steps)))
                self.current_epoch,self.current_step = epoch+1,steps

            # 验证，保存最优模型
            if test_loader and self.args.eval and self.args.test_interval > 0:
                val_loss = self.evalution(test_loader,steps,epoch)

                if self.args.early_stop:
                    
                    acc_num,mn_loss,stop_flag = self.early_stop_wait(self.get_loss_from_val(val_loss),acc_num,mn_loss)
                    if stop_flag:
                        return 
                
            if hasattr(self,'scheduler_G') and self.scheduler_G is not None:
                self.scheduler_G.step()
            if hasattr(self,'scheduler_D') and self.scheduler_D is not None:
                self.scheduler_D.step()
        if self.args.rank == 0 :
            self.vis.close()
            

       
    def early_stop_wait(self,loss,acc_num,mn_loss):

        if self.args.rank == 0:
            if loss < mn_loss:
                mn_loss = loss
                cmd_one = 'cp -r %s %s'%(os.path.join(
                        self.args.checkpoint_path,"%03d-%08d.pth"%(
                                self.current_epoch,self.current_step)),
                                                os.path.join(self.args.checkpoint_path,'final.pth'))
                done_one = subprocess.Popen(cmd_one,stdout=subprocess.PIPE,shell=True)
                done_one.wait()
                acc_num = 0 
            else:
                acc_num += 1
            # 多机多卡，某一张卡退出则终止程序，使用all_reduce
            if self.args.dist:
            
                if acc_num > self.args.stop_interval:
                    signal = torch.tensor([0]).cuda()
                else:
                    signal = torch.tensor([1]).cuda()
        else:
            if self.args.dist:
                signal = torch.tensor([1]).cuda()
        
        if self.args.dist:
            dist.all_reduce(signal)
            value = signal.item()
            if value >= int(os.environ.get("WORLD_SIZE","1")):
                dist.all_reduce(torch.tensor([0]).cuda())
                return acc_num,mn_loss,False
            else:
                return acc_num,mn_loss,True

        else:
            if acc_num > self.args.stop_interval:
                return acc_num,mn_loss,True
            else:
                return acc_num,mn_loss,False

    def run_single_step(self,data,steps):
        data = self.process_input(data)
        self.run_discriminator_one_step(data,steps)
        self.run_generator_one_step(data,steps)

    def select_img(self,data,name='fake',axis=2):
        if data is None:
            return None
        cat_img = []
        for v in data: 
            cat_img.append(v.detach().cpu())
        
        cat_img = torch.cat(cat_img,-1)
        cat_img = torch.cat(torch.split(cat_img,1,dim=0),axis)[0]
       
        return {name:convert_img(cat_img)}



    ##################################################################
    # Helper functions
    ##################################################################

    def get_loss_from_val(self,loss):
        return loss['loss']

    def get_show_inp(self,data):
        if not isinstance(data,list):
            return [data]
        return data

    def use_ddp(self,model,dist=True):
        if model is None:
            return None,None
        if not dist:
            return model,model
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) #用于将BN转换成ddp模式/
        # model = DDP(model,broadcast_buffers=False,find_unused_parameters=True) # find_unused_parameters->训练gan会有判别器或生成器参数不参与训练，需使用该参数
        model = DDP(model,
                    broadcast_buffers=False,
                    # find_unused_parameters=True
                    )
        model_on_one_gpu = model.module #若需要调用self.model的函数，在ddp模式要调用self._model_on_one_gpu
        return model,model_on_one_gpu
    def process_input(self,data):
        
        
        if isinstance(data,list):
            data = [x.to(self.device) for x in data]
        else:
            data = data.to(self.device)
        return data