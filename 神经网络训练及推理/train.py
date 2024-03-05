import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D

import pickle

################################
path = 'D:/Python_Project/OpenCV_Project/neural_metwork_training/train_resources'
pathLabels = 'labels.csv'
testRatio = 0.2
valRatio = 0.2
imageDimensions = (180,180,3)

batchSizeVal = 50
epochsVal = 10
stepsPerEpoch = 2000
################################

images=[]
class_names = []
myList = os.listdir(path)
numofClasses = len(myList)
print("Total Num of Classes Detected",numofClasses)
#print(numofClasses,myList)
print("Importing Classes......")
for x in range(numofClasses):
    myPicList = os.listdir(path+'/'+myList[x])
    #print(myPicList)
    for y in myPicList:
        #print(path+'/'+myList[x]+'/'+y)
        curImg = cv2.imread(path+'/'+myList[x]+'/'+y)
        curImg = cv2.resize(curImg,(imageDimensions[0],imageDimensions[1]))
        images.append(curImg)
        class_names.append(x)
    print(myList[x],end=' ')

images = np.array(images)
class_names = np.array(class_names)
#print(images.shape,class_names.shape)

### Spliting the data
X_train,X_test,y_train,y_test = train_test_split(images,class_names,test_size=testRatio)
X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train,test_size=valRatio)
#print(X_train.shape)
#print(X_test.shape)

numOfSamples = []
for x in range(numofClasses):
    #print(len(np.where(y_train==myList[x])[0]))
    numOfSamples.append(len(np.where(y_train==myList[x])[0]))
#print(numofClasses)
plt.figure(figsize=(10,5))
plt.bar(range(0,numofClasses),numOfSamples)
plt.title("Num Of Images for each Class")
plt.xlabel("Class ID")
plt.ylabel("Num of Images")
plt.show()

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

X_train = np.array(list(map(preProcessing,X_train)))
X_test = np.array(list(map(preProcessing,X_test)))
X_validation = np.array(list(map(preProcessing,X_validation)))

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

y_train = to_categorical(y_train,numofClasses)
y_test = to_categorical(y_test,numofClasses)
y_validation = to_categorical(y_train,numofClasses)

def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNode = 500
    
    model = Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilter1,input_shape=(imageDimensions[0],
                                                            imageDimensions[1],
                                                            1),activation='relu'
                                                            )))
    model.add((Conv2D(noOfFilters,sizeOfFilter1,activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2,sizeOfFilter2,activation='relu')))
    model.add((Conv2D(noOfFilters//2,sizeOfFilter2,activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(noOfNode,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numofClasses,activation='softmax'))
    model.compile(Adam(lr=0.001),loss= 'categorical_crossentropy',metrics=['accuracy'])

    return model

model = myModel()
print(model.summary())



history = model.fit_generator(dataGen.flow(X_train,y_train,
                                 batch_size=batchSizeVal),
                                 steps_per_epoch=stepsPerEpoch,
                                 epochs=epochsVal,
                                 validation_data=(X_validation,y_validation),
                                 shuffle=1)
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('accuracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(X_test,y_test,verbose=0)
print("Test Score = ",score[0])
print("Test Accuracy = ",score[1])



pickle_out = open("D:/Python_Project/OpenCV_Project/neural_metwork_training/train_resources/model_trained.p","wb")
pickle.dump((model,pickle_out))
pickle_out.close()