import os
import re

import numpy as np
import cv2
from matplotlib import pyplot as plt
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

from .rppg.CHROME_DEHAAN import CHROME_DEHAAN, CHROME_DEHAAN_RedBlue
from .rppg.SPO2 import calc_spo2_from_MeanStd, calc_spo2_from_RB
from .rppg.utils import calculate_metric_per_video, calculate_metric_peak_per_video


def preprocess_video(path, type_src='video', fps=30, interval=[10, 30], crop=True, show=False) -> list:    
            
    print('%% crop and reshape')
    #########################################################################
    if type_src == 'video':
        vidObj = cv2.VideoCapture(path)
        totalFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT)) # get total frame size
        i = 0
        def get_next():
            return *vidObj.read(), vidObj.get(cv2.CAP_PROP_POS_MSEC)
    elif type_src == 'images':
        path_imgs = path
        totalFrames = len(path_imgs) # get total frame size
        i = 0
        def get_next():
            success, img = (True, cv2.imread(path_imgs[i])) if i < totalFrames else (False, None)
            time_ms = i * (1000.0 / fps)
            return success, img, time_ms
    else:
        raise Exception('Invalid type')
        
    #########################################################################
    # intialize 
    # face = np.zeros((totalFrames), dtype=bool) # flags
    start, end = interval[0] * fps, interval[1] * fps
    N, H, W, C = end - start, 128, 128, 3
    frames_rPPGNet = np.zeros((N, H, W, C), dtype=np.uint8)
    chicks = {
        'left_chick': [111, 203],
        'right_chick': [349, 411],
        # 'forehead': [67, 296],
    }
    

    #########################################################################
    success, img, t = get_next()
    mean_RGB = []
    std_RGB = []
    
    # mediapipe
    with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
    
        while success:
            
            if i < start:
                success, img, t = get_next()
                i += 1
                continue
            
            if i > (end - 1):
                break
            
            if show:
                tmp_img = img.copy()
                
            # detect and crop
            if crop:
                means = np.zeros(3)
                stds = np.zeros(3)
                # Bm, Gm, Rm = 0, 0, 0
                # Bs, Gs, Rs = 0, 0, 0
                results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # get landmarks
                for face_landmarks in results.multi_face_landmarks:
                    for ltop, rbot in chicks.values():
                        loc_x = int(face_landmarks.landmark[ltop].x * img.shape[1])
                        loc_y = int(face_landmarks.landmark[ltop].y * img.shape[0])
                        loc_xx = int(face_landmarks.landmark[rbot].x * img.shape[1])
                        loc_yy = int(face_landmarks.landmark[rbot].y * img.shape[0])
                        if show:
                            cv2.rectangle(tmp_img, (loc_x, loc_y), (loc_xx, loc_yy), (0,0,255), 2)
                        means = np.mean(img[loc_y:loc_yy, loc_x:loc_xx, :], axis=(0, 1)) + means
                        stds = np.std(img[loc_y:loc_yy, loc_x:loc_xx, :], axis=(0, 1)) + stds
                    hf = int((face_landmarks.landmark[0].y - face_landmarks.landmark[151].y) * img.shape[0])
                    xx, yy = (int(face_landmarks.landmark[197].x * img.shape[1]), int(face_landmarks.landmark[197].y * img.shape[0]))
                face_bbox = [xx - hf, yy - hf, xx + hf, yy + hf]  #  ['ltop_x', 'ltop_y', 'rbot_x', 'rbot_y']
                if show:
                    cv2.rectangle(tmp_img, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (0,0,255), 2)
                mean_RGB.append(means[::-1] / 2)
                std_RGB.append(stds[::-1] / 2)
            else:
                means = np.mean(img, (1, 2))
                stds = np.std(img, (1, 2))
                mean_RGB.append(means[::-1])
                std_RGB.append(stds[::-1])
                face_bbox = []
                
            # resize and save image for rPPGNet
            frames_rPPGNet[i - start] = cv2.resize(img[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]], (H, W))
            # frames_rPPGNet[i - start] = cv2.resize(img, (H, W))
            
            # show frames
            if show:
                cv2.imshow('w1', cv2.resize(tmp_img, (720, 480)))
                # cv2.imshow('w1', cv2.resize(frames_rPPGNet[i - start], (720, 480)))
                if cv2.waitKey(5) == ord('q'):
                    break
            
                        
            # read next image
            success, img, t = get_next()
            i = i + 1
            if i & 0x3f == 0x00:
                print('\r ', i, '/', totalFrames, end="")
        print()    
    
    #########################################################################
    print('%% output length:', len(mean_RGB))
    #########################################################################
    
    # N X C
    return np.asarray(mean_RGB), np.asarray(std_RGB), frames_rPPGNet


def predict_bio_sigs(video_path, fps=60, interval=[10, 20], type_src='video', show=False):
    
    # --------------------------------------
    # Preprocess video
    # --------------------------------------
    mean_RGB, std_RGB, frames_rPPGNet = preprocess_video(
        video_path, fps=fps, interval=interval, type_src=type_src, show=show)

    print('aa', os.getcwd())
    # BP_sig = predict_rPPGNet(frames_rPPGNet, 'sj\\sj_spo2\\rPPGNet_weight.ckpt')
    BP_sig = 0

    # --------------------------
    # generate signals
    # --------------------------
    sig_HR = CHROME_DEHAAN(mean_RGB.copy(), fps)
    sig_RR = CHROME_DEHAAN(mean_RGB.copy(), fps, 'resp')
    sig_red, sig_blue = CHROME_DEHAAN_RedBlue(mean_RGB.copy(), fps)
    sig_spo2 = calc_spo2_from_MeanStd(mean_RGB.copy(), std_RGB)

    # -------------------------------
    # calculate HR, RR, SpO2
    # -------------------------------
    
    HR = calculate_metric_peak_per_video(sig_HR, None, fs=fps)
    RR = calculate_metric_peak_per_video(sig_RR, None, fs=fps, signal='resp')
    
    SpO2 = sig_spo2.mean()
    SBP = BP_sig
    DBP = BP_sig
    
    def set_bound(x, a, b, r=2, integer=True):
        return int(round(max(a, min(x, b)), r)) if integer else round(max(a, min(x, b)), r)
    HR = set_bound(HR, 40., 240., 0)
    RR = set_bound(RR, 5., 60., 0)
    SpO2 = set_bound(SpO2, 70., 100., 1, False)
    SBP = 0
    DBP = 0

    res = {
        'HR': HR,
        'RR': RR,
        'SpO2': SpO2,
        'SBP' : SBP, # SBP,
        'DBP' : DBP, # DBP,
    }
    return res


def predict_label_sigs(ppg, fs):
    
    HR = calculate_metric_per_video(ppg, None, fs=fs)
    RR = calculate_metric_per_video(ppg, None, fs=fs, signal='resp')
    
    res = {
        'HR': HR,
        'RR': RR,
    }
    return res


def test():
    video_path = 'vid.mp4'
    res = predict_bio_sigs(video_path, 60, [5, 10], show=True)
    print(res)


if __name__ == '__main__':
    test()

