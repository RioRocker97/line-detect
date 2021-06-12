from yolo_detect import prepareYolo,runYolo
import torch
import subprocess
import cv2
if __name__ == "__main__":
    subprocess.call('cls',shell=True)
    print("Run YOLOv5...")
    prepareYolo('yolov5m.pt',confidence=0.5,loadFromImage=True,imageSource='response.jpg')
    res = runYolo()
    cv2.imwrite('res.jpg',res)