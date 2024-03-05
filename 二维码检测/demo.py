import cv2
import numpy as np
from pyzbar.pyzbar import decode

#img = cv2.imread("1.png")
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    success,img = cap.read()
    for barcode in decode(img):
        print(barcode.rect)
        mydata = barcode.data.decode('utf-8')
        print(mydata)
        pts = np.array([barcode.polygon],np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,(255,255,0),5)
        pts2 = barcode.rect
        cv2.putText(img,mydata,(pts2[0],pts2[1]),cv2.FONR_HERSHEY_SIMPLE,
                    0.9,(255,255,0),2)
    cv2.imshow("result",img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
