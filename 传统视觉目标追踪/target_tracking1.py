import cv2
import numpy as np
video = cv2.VideoCapture("D:/video/tanker_game.mp4")
x1,y1,x2,y2=0,0,0,0


def select_object(event,x,y,flags,param):
    global x1,y1,x2,y2
    if event==cv2.EVENT_LBUTTONDOWN:
        x1,y1 = x,y
        x2,y2 = x,y
    elif event == cv2.EVENT_LBUTTONUP:
        x2,y2 = x,y
    x1,x2 = min(x1,x2),max(x1,x2)
    y1,y2 = min(y1,y2),max(y1,y2)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2)
    cv2.imshow("SelectObject",frame)


ret1,frame = video.read()
cv2.namedWindow('SelectObject')
cv2.setMouseCallback("SelectObject",select_object)
cv2.waitKey()
while True:
    ret,image = video.read()
    image1 = frame[y1:y2,x1:x2]
    gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edge1 =cv2.Canny(image1,200,250)
    edge = cv2.Canny(image,200,250)
    contours1,hierarchy1 = cv2.findContours(edge1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for con in contours1:
        area = cv2.contourArea(con)
        if area > max_area:
            max_area = area
            max_con = con
    contours,hierarchy = cv2.findContours(edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for con in contours:
        ret = cv2.matchShapes(cv2.UMat(max_con),con,1,0.0)
        if ret < 0.05:
            x,y,w,h = cv2.boundingRect(con)
            draw = np.array([[[x,y]],[[x+w,y]],[[x+w,y+h]],[[x,y+h]]])
            cv2.drawContours(image,[draw],-1,(255,255,0),2)
    cv2.imshow("edge1",edge1)
    cv2.imshow("edge1",edge1)
    cv2.imshow("image",image)
    if cv2.waitKey(1) == ord(' '):
        break
video.release()
cv2.destroyAllWindows()
