import cv2
import numpy as np
from pyzbar.pyzbar import decode

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

with open('mydatafile') as f:
    mydatalist=f.read().splitlines()

while True:
    #success,img = cap.read()
    img = cv2.imread("1.png")
    for barcode in decode(img):
        #print(barcode.rect)
        mydata = barcode.data.decode('utf-8')
        print(mydata)
        if mydata in mydatalist:
            myoutput='Authorized'
            mycolor = (0,255,0)
        else:
            myoutput='Un-Authorized'
            mycolor = (0,0,255)
        pts = np.array([barcode.polygon],np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,mycolor,5)
        pts2 = barcode.rect
        cv2.putText(img,myoutput,(pts2[0],pts2[1]),cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,mycolor,2)
    cv2.imshow("result",img)
    if cv2.waitKey(1) & 0xFF == 27:
        break