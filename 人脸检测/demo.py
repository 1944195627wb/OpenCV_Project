import cv2
import numpy as np
import face_recognition

img1 = face_recognition.load_image_file('images/1.png')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2 = face_recognition.load_image_file('images/2.png')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(img1)[0]
encodeElon = face_recognition.face_encodings(img1)[0]
cv2.rectangle(img1,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,255,0),2)

faceLocTest = face_recognition.face_locations(img2)[0]
encodeTest = face_recognition.face_encodings(img2)[0]
cv2.rectangle(img2,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(0,255,0),2)

results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)
cv2.putText(img2,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Elon Musk',img1)
cv2.imshow('Elon Test',img2)
cv2.waitKey(0)