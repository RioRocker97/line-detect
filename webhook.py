from flask import Flask, request, Response,send_file
from yolo_detect import prepareYolo,runYolo
from io import BytesIO
import torch,subprocess,cv2,os,json,certifi,time,urllib3
app = Flask(__name__)
access_token = "vynoK6xqJeqGeBqSWPO5E1p0H2PNCgKyfBO2xwNshzWvK5Dh8IeKIn9G4mQJyO/q9+2ElnqOcp4iRvsN+3QtiRscMdJkgcolppMYqYsvg30p+PERkxMRGdJ+xog8jD0Lc1lvtgothprXx4zLmg4mhgdB04t89/1O/w1cDnyilFU="
# will do access_token encryption later 
####################
HTTP = urllib3.PoolManager()
####################
def get_webhook_endpoint():
    rep = HTTP.request('GET','https://api.line.me/v2/bot/channel/webhook/endpoint',headers={
        'Content-Type': 'application/json',
        'Authorization': 'Bearer '+ access_token
    })
    if str(rep.status) == '200':
        body = json.loads(rep.data.decode('UTF-8'))['endpoint']
        return body
    else:
        return "ERROR"
def get_img(id):
    rep = HTTP.request('GET','https://api-data.line.me/v2/bot/message/'+id+'/content',headers={
        'Authorization': 'Bearer '+ access_token
    })
    if str(rep.status) == '200':
        img = open('raw.jpg','wb')
        img.write(rep.data)
        img.close()
    else:
        print('Error At Get-IMG',rep.status)
def text_reply(reply_token,user_id="123",msg_type="text",msg="This reply came from developer's Laptop"):
    data = json.dumps({
        "replyToken": reply_token,
        "messages":[
            {
                "type":"text",
                "text":msg
            }
        ]
    }).encode('UTF-8')

    rep = HTTP.request('POST','https://api.line.me/v2/bot/message/reply',body=data,headers={
        'Content-Type': 'application/json',
        'Authorization': 'Bearer '+ access_token
    })
    if str(rep.status) == '200':
        print("Reply Ok")
    else :
        print("Reply Error ", rep.status)
def detect_reply(user_id):

    num = len(os.listdir('./result'))+1
    prepareYolo('yolov5s.pt',confidence=0.4,loadFromImage=True,imageSource='raw.jpg')
    res = runYolo()
    cv2.imwrite('./result/res'+str(num)+'.jpg',res)

    webhook = get_webhook_endpoint()
    data = json.dumps({
        "to": user_id,
        "messages":[
            {
                "type":"image",
                "originalContentUrl": webhook+'/res/res'+str(num)+'.jpg',
                "previewImageUrl": webhook+'/res/res'+str(num)+'.jpg'
            }
        ]
    }).encode('UTF-8')

    rep = HTTP.request('POST','https://api.line.me/v2/bot/message/push',body=data,headers={
        'Content-Type': 'application/json',
        'Authorization': 'Bearer '+ access_token
    })
    if str(rep.status) == '200':
        print('PUSH ok')
    else:
        print ('PUSH error',rep.status)
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
            if payload['type'] == 'image':
                user_id = address['userId']
                img_id = payload['id']
                print("LINE image ID : ",img_id)
                text_reply(token,msg="Processing.... Please Wait")
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