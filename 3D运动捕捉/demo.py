import cv2
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture('ikun.mp4')
posList = []
detector = PoseDetector()
while True:
    success,img = cap.read()
    img = detector.findPose(img)
    lmList,bboxInfo = detector.findPosition(img)
    if bboxInfo:
        #lm由四个坐标x,y,z构成
        lmString = ''
        for lm in lmList:
            #print(lm)
            lmString += f'{lm[0]},{img.shape[0]-lm[1]},{lm[2]},'
        posList.append(lmString)
        print(len(posList))
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        with open("AnimationFile.txt",'w') as f:
            f.writelines(["%s\n" % item for item in posList])
    elif key==ord("q"):
        break