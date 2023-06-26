import numpy as np
import cv2
import torch

class FaceParsing:
    def __init__(self,model_path='pretrained_models/face_parsing.pt'):
        self.parsing_model = torch.jit.load(model_path)
        if torch.cuda.is_available():
            self.parsing_model = self.parsing_model.cuda()
        self.parsing_model.eval()
        self.mean =np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1,3,1,1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1,3,1,1)
       

    def get_parsing(self,img):
        '''
        atts = ['background-0','skin-1', 'l_brow-2', 'r_brow-3', 'l_eye-4', 'r_eye-5', 'eye_g-6', 'l_ear-7', 'r_ear-8', 'ear_r-9',
            'nose-10', 'mouth-11', 'u_lip-12', 'l_lip-13', 'neck-14', 'neck_l-15', 'cloth-16', 'hair-17', 'hat-18']
        '''
        return self.parsing_model(img)

    def preprocess_parsing(self,input_image_np):
        input_image_np = cv2.resize(input_image_np, (512,512))
        input_image_np = input_image_np.transpose((2,0,1)) / 255.
        input_image_np = np.array(input_image_np[np.newaxis, :])
    
        input_image_np = (input_image_np[:,0,...] - self.mean) / self.std
       
    
        input_image_np = torch.from_numpy(input_image_np.astype(np.float32))
        if torch.cuda.is_available():
            input_image_np = input_image_np.cuda()
        return input_image_np

    def postprocess_parsing(self,y,h=None,w=None):
        y = y[0].argmax(0).cpu().numpy()
        if h is not None:
            y = cv2.resize(y.astype(np.uint8),[w,h],interpolation=cv2.INTER_NEAREST)
        return y