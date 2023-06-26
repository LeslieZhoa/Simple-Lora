'''
@author LeslieZhao
@date 20230620
'''

import torch
import pdb 
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import argparse
import os
import numpy as np
import cv2
parser = argparse.ArgumentParser(description="infer")

parser.add_argument('--basemodel',default='pretrained_models/chilloutmixNiPruned_Tw1O',type=str,help='base model path')
parser.add_argument('--lora_path',default=None,type=str,help='lora model path')
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


if __name__ == "__main__":
    device = 'cuda'
    args = parser.parse_args()
    pipeline = StableDiffusionPipeline.from_pretrained(args.basemodel,safety_checker=None).to(device)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config, use_karras_sigmas=True
    )

    lora_state = torch.load(args.lora_path)['lora']
    pipeline.load_lora_weights(lora_state)

    pipeline.safety_checker = lambda images, clip_input: (images, None)

    images = pipeline(prompt=args.prompt, 
        negative_prompt=args.neg_prompt, 
        width=args.width, 
        height=args.height, 
        num_inference_steps=args.num_inference_steps, 
        num_images_per_prompt=args.num_images_per_prompt,
        generator=torch.manual_seed(args.seed),
        guidance_scale=args.guidance_scale,
        cross_attention_kwargs={"scale": args.scale},
        output_type='np'
    ).images

    images = np.concatenate(images,1)
    images = np.clip(images*255.,0,255).astype(np.uint8)
    images = cv2.cvtColor(images,cv2.COLOR_RGB2BGR)
    save_base = os.path.split(args.outpath)[0]
    os.makedirs(save_base,exist_ok=True)
    cv2.imwrite(args.outpath,images)
    