import cv2
import numpy as np
from datetime import datetime
cap = cv2.VideoCapture10)
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3

classesFile='coco.names'
classNames=[]
with open(classesFile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')
# print(classNames)
# print(len(classNames))
#大模型
# modelConfiguration= 'yolov3.cfg'
# modelWeights = 'yolov3.weights'
#小模型
modelConfiguration= 'yolov3-tiny.cfg'
modelWeights = 'yolov3-tiny.weights'
#加载模型
net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
#指定使用CPU
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# 指定使用CUDA后端和目标
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

def findObjects(outputs,img):
    hT,wT,cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT),int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2),int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    #print(len(bbox))
    print(bbox)
    print(confs)
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    resultNames = [classNames[i] for i in classIds]
    print(resultNames)

    for i in indices:
        #i= i[0]
        #print(i)
        box = bbox[i]
        box =bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()}{int(confs[i]*100)}%',
                    (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)



while True:
    success, img = cap.read()
    start = datetime.now()

    blob = cv2.dnn.blobFromImage(img, 1/255, (whT,whT), [0,0,0],1, crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    #print(layerNames)
    #print(net.getUnconnectedOutLayers())
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)

    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)

    findObjects(outputs,img)
    end =datetime.now()
    time = end-start
    time=int(1/time.total_seconds())
    print(f'FPS={time}')

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()