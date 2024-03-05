import cv2
import math
import numpy as np

path = 'test.png'
img = cv2.imread(path)
pointsList=[]

def mousePoints(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        size = len(pointsList)
        if size != 0 and size % 3 != 0:
            cv2.line(img,tuple(pointsList[round((size-1)/3)*3]),(x,y),(0,0,255),2)
        cv2.circle(img,(x,y),5,(0,0,255),cv2.FILLED)
        pointsList.append([x,y])

# #按照斜率计算公式
# def gradient(pt1,pt2):
#     return (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
# def getAngle(pointsList):
#     pt1,pt2,pt3 = pointsList[-3:]
#     k1 = gradient(pt1,pt2)
#     k2 = gradient(pt1,pt3)
#     angle = math.atan((k2-k1)/(1+k1*k2))
#     angle = round(math.degrees(angle))
#     cv2.putText(img,str(angle),(pt1[0]-40,pt1[1]-20),cv2.FONT_HERSHEY_COMPLEX,1.5,(0,0,255),2)
#     return angle

#按照余弦定理计算公式
def distance(pt1,pt2):
    d = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
    return d
def getAngle(pointsList):
    pt1, pt2, pt3 = pointsList[-3:]
    d1 = distance(pt1,pt2)
    d2 = distance(pt1,pt3)
    d3 = distance(pt2,pt3)
    angle = int(np.degrees(np.arccos((d1*d1+d2*d2-d3*d3)/(2*d1*d2))))
    cv2.putText(img, str(angle), (pt1[0] - 40, pt1[1] - 20), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
    return angle

while True:
    if len(pointsList) % 3==0 and len(pointsList) !=0:
        angle = getAngle(pointsList)
        print(angle)
    cv2.imshow('Image',img)
    cv2.setMouseCallback('Image',mousePoints)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    if cv2.waitKey(1) & 0xFF ==ord(' '):
        pointsList = []
        img = cv2.imread(path)

cv2.destroyAllWindows()