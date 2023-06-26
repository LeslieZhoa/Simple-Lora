import argparse
import os
parser = argparse.ArgumentParser(description="blip")

parser.add_argument('--img_base',default=None,help='input image base path')
parser.add_argument('--ori_txt',default=None,type=str,help='input image base path')
parser.add_argument('--new_txt',default=None,type=str,help='input image base path')

def change_text(txt_paths,ori_txt,new_txt):
    for i,txt_path in enumerate(txt_paths):
        with open(txt_path,'r') as f:
            info = f.read().strip().replace(ori_txt,new_txt)
        with open(txt_path,'w') as f:
            f.write(info)
        print('\rhave done %04d'%i,end='',flush=True)
    print()
    print('Done!!!')
    
if __name__ == "__main__":
    fn = lambda x:[os.path.join(x,f) for f in os.listdir(x) if  f.endswith('.txt')]
    args = parser.parse_args()
    change_text(fn(args.img_base),args.ori_txt,args.new_txt)
    