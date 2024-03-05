import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras .utils.np_utils import to_categorical
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras_preprocessing.image import ImageDataGenerator


###########Parameters############
path = "myData"
labelFile = 'labels.csv'
batch_size_val=50
steps_per_epoch_val=2000
epochs_val=30
imageDimesions=(32,32,3)
testRatio = 0.2
vallidationRatio = 0.2


##########################
count = 0
images= []
classNo = []
myList = os.listdir(path)
print("total classes Detected:",len(myList))
noOfClasses = len(myList)
print("Import Classes.....")
for x in range(len(myList)):
    myPicList = os.listdir(path+'/'+str(count))
    for y in myPicList:
        curImg = cv2.imread(path+"/"str(count)+"/"+y)
        images.append(curImg)
        classNo.append(count)
    print(count,end=" ")
    count+=1
print(" ")
images = np.array(images)
classNo = np.array(classNo)
#####################
x_train,x_test,y_train,y_test = train_test_split(images,classNo,test_size=testRatio)
x_train,x_validation,y_train,y_validation = train_test_split(x_train,y_train,test_size=vallidationRatio)


######################
print("Data Shapes")
print("Train",end = "");print(x_train.shape,y_train.shape)
print("Validation",end='');print(x_validation.shape,y_validation.shape)
print("test",end="");print(x_test.shape,y_test.shape)
assert(x_train.shape[0]==y_train.shape[0])
assert(x_validation.shape[0]==y_validation.shape[0])
assert(x_test.shape[0]==y_test.shape[0])
assert(x_train.shape[1]==y_train.shape[1])
assert(x_validation.shape[1]==y_validation.shape[1])
assert(x_test.shape[1]==y_test.shape[1])

data = pd.read_csv(labelFile)
print('data shape',data.shape,type(data))
num_of_samples = []
cols = 5
num_classes = noOfClasses
fig,axs = plt.subplots(nrows=num_classes,ncols=cols,figsize=(5,300))
fig.tight_layout()
for i in range(cols):
    for j,row in data.iterrows():
        x_selected = x_train[y_train==j]
        axs[j][i].imshow(x_selected[random.randint(0,len(x_selected)-1),:,:],cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i ==2:
            axs[j][i].set_title(str(j)+"-"row["Name"])
            num_of_samples.append(len(x_selected))



print(num_of_samples)
plt.figure(figsize=(12,4))
plt.bar(range(0,num_classes),num_classes)
plt.xlabel("Class number")
plt.ylabel("number of images")
plt.show()


def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
#############################################
x_train = np.array(list(map(preprocessing,x_train)))
x_validation = np.array(list(map(preprocessing,x_validation)))
x_test = np.array(list(map(preprocessing,x_test)))
cv2.imshow("GrayScale Images",x_train[random.randint(0,len(x_train)-1)])


#############################################
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_validation = x_validation.reshape(x_validation.shape[0],x_validation[1],x_val)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

#############################################
dataGen = ImageDataGenerator(width_shift_range=0.1
                             height_shift_range=0.1
                             zoom_range=0.2
                             shear_range=0.1
                             rotation_range=10
                             )
dataGen.fit(x_train)
batches=dataGen.flow(x_train,y_train,batch_size=20)
x_batch,y_bathch = next(batches)


fig,axs = plt.subplots(1,15,figsize=(20,5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(x_batch[i].reshpae(imageDimesions[0],imageDimesions[1]))
    axs[i].axis('off')
plt.show()

y_train = to_categorical(y_train,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)


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
    model.add(Dense(noOfClasses,activation='softmax'))
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
plt.plot(history.histroy['loss'])
plt.plot(history.histroy['val_loss'])
plt.legend('training','validation')
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.histroy['loss'])
plt.plot(history.histroy['val_loss'])
plt.legend('training','validation')
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()

score = model.evaluate(x_test,y_test,verbose=0)
print("Test Score = ",score[0])
print("Test Accuracy = ",score[1])   

pickle_out = open("D:/Python_Project/OpenCV_Project/交通标志分类/model_trained.p","wb")
pickle.dump((model,pickle_out))
pickle_out.close()