import os
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load,Ensemble
from utils.datasets import LoadStreams,LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, strip_optimizer, set_logging)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import threading
import subprocess

## Global Variable ###
dataset = 0
model = Ensemble()
colors = 0
names = 0
device = 0
half = False
new_unk = False
imgsz = 320
onlyOne = False
conf = 0
#####################
def prepareYolo(model_path,confidence=0.5,loadFromImage=False,imageSource=''):
    global dataset,model,colors,names,device,half,imgsz,onlyOne,conf

    weights = model_path
    onlyOne = loadFromImage
    conf = confidence
    if(torch.cuda.device_count() == 0):
        print('Using CPU')
        device = select_device('cpu')
    else:
        print('Using GPU : '+torch.cuda.get_device_name(0))
        device = select_device('0')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    cudnn.benchmark = True  # set True to speed up constant image size inference
    if onlyOne :
        dataset = LoadImages(imageSource, img_size=imgsz)
    else :
        dataset = LoadStreams('0', img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
def runYolo():
    global dataset,model,colors,names,device,half,new_unk,onlyOne,conf

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    ################# Preparation ##########################################
    dataset.__iter__()
    path,img,im0s,vid_cap = dataset.__next__()
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized() #start predictiong
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred,conf_thres=conf,iou_thres=0.45)
    t2 = time_synchronized()
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if onlyOne :
            p, s, im0 = path[i], '%g: ' % i, im0s
        else :
            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()

        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        max_count=0
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string
                if(names[int(c)] != 'Unknown'):
                    max_count+=n

            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #line = (cls, conf, *xywh) if opt.save_conf else (cls, *xywh)  # label format . comment it out for lazy implement
                line = (cls,*xywh)

                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                print()
                if onlyOne :
                    print("Found : %s at %.2f %.2f %.2f %.2f " % (names[int(cls)],line[1],line[2],line[3],line[4]))

        # Stream results
        if not onlyOne :
            cv2.destroyWindow('YOLO')

    print("Time To Detect : %.2f" % float(time.time() - t0))
    return im0


