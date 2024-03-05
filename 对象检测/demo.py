import cv2
import numpy as np
cap = cv2.VideoCapture(0)
thres = 0.45
nms_threshold=0.2
#长度
cap.set(3, 1400)
#宽度
cap.set(4, 1200)

classNames = []
classFile = 'coco.names'

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320, 230)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
# clssIds: the returned ids , confs: the confidence , bbox: the drawn box
while True:
    ret, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    #print(classIds, bbox)
    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
    for id,confidence in zip(indices,confs):
        box = bbox[id]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
        cv2.putText(img, classNames[classIds[id]-1]+f'  {confidence:.2f}', (box[0] + 10, box[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 0), thickness=2)

    # if len(classIds) != 0:
    #     for classid, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
    #         cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
    #         cv2.putText(img, classNames[classid - 1]+f'  {confidence:.2f}', (box[0] + 10, box[1] + 20),
    #         cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 0), thickness=2)

    cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()