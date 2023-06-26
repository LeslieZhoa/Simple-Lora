from models.blip import blip_decoder
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import torch
import os
import pdb
import argparse

parser = argparse.ArgumentParser(description="blip")

parser.add_argument('--img_base',default=None,help='input image base path')

class Process:
    def __init__(self,model_path):
        self.model = blip_decoder(pretrained=model_path, image_size=384, vit='base')
        self.model.eval()
        self.model.cuda()
        self. transform = transforms.Compose([
        transforms.Resize((384,384),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 

    def __call__(self,img_paths):
        for i,img_path in enumerate(img_paths):
            self.forward(img_path)
            print('\rhave done %04d'%i,end='',flush=True)

        print()
        print('Done!!!')

    def forward(self,img_path):
        img = Image.open(img_path).convert('RGB')  
        inp = self.transform(img).unsqueeze(0).cuda()
        with torch.no_grad():
            caption = self.model.generate(inp, sample=False, num_beams=3, max_length=48, min_length=24)[0]
        txt_path = '.'.join(img_path.split('.')[:-1]) + '.txt'
        with open(txt_path,'w') as f:
            f.write(caption) 

if __name__ == "__main__":
    fn = lambda x:[os.path.join(x,f) for f in os.listdir(x) if not f.endswith('.txt')]
    process = Process('../../pretrained_models/model_base_caption_capfilt_large.pth')
    args = parser.parse_args()
    process(fn(args.img_base))