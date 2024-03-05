import numpy as np
import cv2
import pickle
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img
#################################
width =640
height = 480
threshold=0.8
#################################
cap=cv2.VideoCapture(1)
cap.set(3,width)
cap.set(4,height)

pickle_in = open('model_trained.p','rb')
model= pickle.load(pickle_in)
class_names = ['1','2','3','A','a','B','b','C','c','D','d','E','e','F','f','G','g','H','h','I','i','J','j','K','k','L','l','M','m','N','n','O','o']
while True:
    success,imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(180,180))
    img = preProcessing(img)
    #cv2.imshow('Processed Image',img)
    img = img.reshape(1,180,180,1)
    #预测
    classIndex = int(model.predict_classes(img))
    predictions = model.predict(img)
    probVal = np.amax(predictions)

    if probVal>threshold:
        cv2.putText(imgOriginal,class_names[classIndex]+'   '+str(probVal),
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)
    cv2.imshow("Original Image",imgOriginal)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
