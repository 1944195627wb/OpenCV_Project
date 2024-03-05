import cv2
import numpy as np
import dlib
from datetime import datetime
cap = cv2.VideoCapture(0)
while True:
    success,img = cap.read()
    start = datetime.now()
    #img = cv2.resize(img,(0,0),None,0.5,0.5)
    imgOriginal = img.copy()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)
    for face in faces:
        x1,y1 = face.left(),face.top()
        x2,y2 = face.right(),face.bottom()
        imgOriginal = cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        landmarks = predictor(imgGray,face)
        for n in range(0,68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(imgOriginal,(x,y),2,(0,255,0),cv2.FILLED)
    end = datetime.now()
    print("FPS: ",f"{1/(end-start).total_seconds():.2f}")
    cv2.imshow("Original Image",imgOriginal)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
