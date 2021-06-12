from flask import Flask, request, Response
from yolo_detect import prepareYolo,runYolo
import torch
import subprocess
import cv2
app = Flask(__name__)

@app.route('/', methods=['POST'])
def respond():
    print(request.json);
    return Response(status=200)
@app.route('/',methods=['GET'])
def index():
    return "Webhook 1.0"

if __name__ == "__main__":
    """
    subprocess.call('cls',shell=True)
    print("Run Webhook...")
    prepareYolo('yolov5s.pt',confidence=0.4,loadFromImage=True,imageSource='meow_irl.jpg')
    res = runYolo()
    cv2.imwrite('res.jpg',res)
    """
    app.run(debug=True,host='0.0.0.0',port=80)