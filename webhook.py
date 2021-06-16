from flask import Flask, request, Response,send_file
from yolo_detect import prepareYolo,runYolo
from io import BytesIO
import torch,subprocess,cv2,os,pycurl,json,certifi,time
app = Flask(__name__)
access_token = "m2QfiJ4wsiIYdCgooSSbxzdO/PqjQXZ1//G1wwKRfPhNW/vieinIHB2Y7ouVW3NVXoMkdOKyHCeJaM5g2eXjKWLNGPaumDOjWugd1+VxlLh65Q4ZM0VunHxjlxErRjXQp9mapvzudJ6yJMZ/qImobgdB04t89/1O/w1cDnyilFU="
# will do access_token encryption later 
####################
def get_webhook_endpoint():
    reply = pycurl.Curl()
    rep = BytesIO()
    reply.setopt(pycurl.URL,'https://api.line.me/v2/bot/channel/webhook/endpoint')
    reply.setopt(pycurl.CAINFO,certifi.where())
    reply.setopt(pycurl.HTTPHEADER,[
        'Content-Type: application/json',
        'Authorization: Bearer '+ access_token
    ])
    reply.setopt(pycurl.WRITEDATA,rep)
    reply.perform()

    if(str(reply.getinfo(pycurl.HTTP_CODE)) == '200'):
        body = json.loads(rep.getvalue())['endpoint']
        return body
    else:
        return "error"
def get_img(id):
    reply = pycurl.Curl()
    rep = BytesIO()
    reply.setopt(pycurl.URL,'https://api-data.line.me/v2/bot/message/'+id+'/content')
    reply.setopt(pycurl.CAINFO,certifi.where())
    reply.setopt(pycurl.HTTPHEADER,[
        'Authorization: Bearer '+ access_token
    ])
    reply.setopt(pycurl.WRITEDATA,rep)
    reply.perform()

    if(str(reply.getinfo(pycurl.HTTP_CODE)) == '200'):
        img = open('raw.jpg','wb')
        img.write(rep.getvalue())
        img.close()
        
    else:
        print("error")
def text_reply(reply_token,user_id="123",msg_type="text"):
    data = json.dumps({
        "replyToken": reply_token,
        "messages":[
            {
                "type":"text",
                "text":"Hello from ur shitty PC"
            }
        ]
    })

    reply = pycurl.Curl()
    reply.setopt(pycurl.URL,'https://api.line.me/v2/bot/message/reply')
    reply.setopt(pycurl.POST,1)
    reply.setopt(pycurl.CAINFO,certifi.where())
    reply.setopt(pycurl.HTTPHEADER,[
        'Content-Type: application/json',
        'Authorization: Bearer '+ access_token
    ])
    reply.setopt(pycurl.POSTFIELDS,data)
    reply.perform()
    if(str(reply.getinfo(pycurl.HTTP_CODE)) == '200'):
        print("Reply Ok")
    else :
        print("Reply Error ",reply.getinfo(pycurl.HTTP_CODE))
    reply.close()
def detect_reply(user_id):

    prepareYolo('yolov5s.pt',confidence=0.4,loadFromImage=True,imageSource='raw.jpg')
    res = runYolo()
    cv2.imwrite('./result/res.jpg',res)

    webhook = get_webhook_endpoint()
    data = json.dumps({
        "to": user_id,
        "messages":[
            {
                "type":"image",
                "originalContentUrl": webhook+'/res/res.jpg',
                "previewImageUrl": webhook+'/res/res.jpg'
            }
        ]
    })
    reply = pycurl.Curl()
    reply.setopt(pycurl.URL,'https://api.line.me/v2/bot/message/push')
    reply.setopt(pycurl.POST,1)
    reply.setopt(pycurl.CAINFO,certifi.where())
    reply.setopt(pycurl.HTTPHEADER,[
        'Content-Type: application/json',
        'Authorization: Bearer '+ access_token
    ])
    reply.setopt(pycurl.POSTFIELDS,data)
    reply.perform()

    if(str(reply.getinfo(pycurl.HTTP_CODE)) == '200'):
        print("Push Ok")
    else :
        print("Push Error ",reply.getinfo(pycurl.HTTP_CODE))
####################


@app.route('/', methods=['POST'])
def respond():
    line_events = request.json['events']
    for event in line_events:
        if event['type']=='message':
            token = event['replyToken']
            address = event['source']
            payload = event['message']
            if payload['type'] == 'text':
                print("from : ",address['userId'])
                print("Text msg : ",payload['text'])
                text_reply(token)
                time.sleep(3)
                detect_reply(address['userId'])
            if payload['type'] == 'image':
                user_id = address['userId']
                img_id = payload['id']
                print("LINE image ID : ",img_id)
                get_img(img_id)
                detect_reply(user_id)

    return Response(status=200)
@app.route('/',methods=['GET'])
def index():
    return "Webhook 1.0"
@app.route('/res/<file_name>',methods=['GET'])
def get_res(file_name):
    return send_file('./result/'+file_name)
if __name__ == "__main__":
    subprocess.call('cls',shell=True)
    app.run(debug=True,host='0.0.0.0',port=80)