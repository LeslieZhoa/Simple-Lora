#! /usr/bin/python 
# -*- encoding: utf-8 -*-
'''
@author LeslieZhao
@date 20230620
'''
from torchvision import transforms 
from transformers import CLIPTokenizer
from dataloader.DataLoader import DatasetBase
from diffusers import StableDiffusionPipeline
import random
import math
import numpy as np
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import os


class DiffusersData(DatasetBase):
    def __init__(self, slice_id=0, slice_count=1,dist=False, **kwargs):
        super().__init__(slice_id, slice_count,dist, **kwargs)
        
        self.transform = transforms.Compose(
        [
            transforms.Resize(kwargs['resolution'], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(kwargs['resolution']) if kwargs['center_crop'] else transforms.RandomCrop(kwargs['resolution']),
            transforms.RandomHorizontalFlip() if kwargs['random_flip'] else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
         )
        
        # self.tokenizer = CLIPTokenizer.from_pretrained(
        #     kwargs['pretrained_model_name_or_path'], subfolder="tokenizer", revision=kwargs['revision']
        #     )
        self.pipeline = StableDiffusionPipeline.from_pretrained(kwargs['basemodel'],safety_checker=None)
       
        self.tokenizer = self.pipeline.tokenizer
        self.custom = kwargs['custom']
        if not self.custom:
            dataset = load_dataset(
                kwargs['dataset_name']
            )
            self.img_column = kwargs['img_column']
            self.caption_column = kwargs['caption_column']
            self.train_data = dataset['train']
            dis1 = math.floor(len(self.train_data)/self.count)
            self.train_data = self.train_data[self.id*dis1:(self.id+1)*dis1]
            
            self.length = len(self.train_data[self.img_column])
            
        else:
            base = kwargs['base']
            self.paths = [os.path.join(base,f) for f in os.listdir(base) if not f.endswith('.txt')]
            dis1 = math.floor(len(self.paths)/self.count)
            self.paths = self.paths if len(self.paths[self.id*dis1:(self.id+1)*dis1]) == 0 else self.paths[self.id*dis1:(self.id+1)*dis1]
            random.shuffle(self.paths)
            self.length = len(self.paths)
            # self.length = 1000
        self.test_caption = self.get_test_caption()

    def __getitem__(self,i):
        if self.custom:
            img_path = self.paths[i%self.length]
            txt_path = '.'.join(img_path.split('.')[:-1])+'.txt'
            with open(txt_path) as f:
                caption = f.read().strip()
        else:
            img_path = BytesIO(self.train_data[self.img_column][i%self.length]['bytes'])
            caption = self.train_data[self.caption_column][i%self.length]
        
        img = Image.open(img_path)
        img = self.transform(img.convert('RGB'))
        input_ids = self.tokenize_captions(caption)
        return input_ids,img

    def get_test_caption(self):
        i = random.choice(range(self.length))
        if self.custom:
            img_path = self.paths[i%self.length]
            txt_path = '.'.join(img_path.split('.')[:-1])+'.txt'
            with open(txt_path) as f:
                caption = f.read().strip()
        else:
            img_path = BytesIO(self.train_data[self.img_column][i%self.length]['bytes'])
            caption = self.train_data[self.caption_column][i%self.length]

        return caption
    
    def tokenize_captions(self,caption):
       
        if isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
            caption = random.choice(caption)
            
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids.squeeze(0)

    def __len__(self):
        # return self.length
        return 2000

