import cv2
import time
import PoseDetectionModule as pm


cap = cv2.VideoCapture('ikun.mp4')
pTime = 0

while True:
    success, img = cap.read()

    detector = pm.PoseDetector()
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    #print(lmList)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

