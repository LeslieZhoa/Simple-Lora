# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from model.third.openpose import util
from model.third.openpose.body import Body
from model.third.openpose.hand import Hand
import cv2


class OpenposeDetector:
    def __init__(self,
                 body_modelpath='pretrained_models/openpose/body_pose_model.pth',
                 hand_modelpath='pretrained_models/openpose/hand_pose_model.pth'):
       
        self.body_estimation = Body(body_modelpath)
        self.hand_estimation = Hand(hand_modelpath)

    def get_pose(self, oriImg, hand=False,detect_resolution=512,image_resolution=512):
        
        oriImg = self.pose_preprocess(oriImg,detect_resolution=detect_resolution)
        with torch.no_grad():
            candidate, subset = self.body_estimation(oriImg)
            canvas = np.zeros_like(oriImg)
            canvas = util.draw_bodypose(canvas, candidate, subset)
            if hand:
                hands_list = util.handDetect(candidate, subset, oriImg)
                all_hand_peaks = []
                for x, y, w, is_left in hands_list:
                    peaks = self.hand_estimation(oriImg[y:y+w, x:x+w, :])
                    peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
                    peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
                    all_hand_peaks.append(peaks)
                canvas = util.draw_handpose(canvas, all_hand_peaks)
        detected_map = self.pose_postprocess(canvas,image_resolution=image_resolution)
        
        return detected_map, dict(candidate=candidate.tolist(), subset=subset.tolist())
        
    def pose_preprocess(self,input_image,detect_resolution=512):
        input_image = input_image.astype(np.float32)
        input_image = util.resize_image(input_image, detect_resolution)
        return input_image

    def pose_postprocess(self,detected_map,image_resolution=512):
        
        detected_map = cv2.resize(detected_map, (image_resolution, image_resolution), interpolation=cv2.INTER_NEAREST)
        detected_map = detected_map.astype(np.uint8)
        return cv2.cvtColor(detected_map,cv2.COLOR_BGR2RGB)
