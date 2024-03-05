import cv2
import numpy as np
import utlis

#图片和其宽高
path='test.png'
widthImg = 700
heightImg = 700
questions = 5
choices = 5
ans = [1,2,0,1,4]
webcamFeed =False
cap = cv2.VideoCapture(1)
cap.set(10,150)
while True:
    if webcamFeed:
        success,img = cap.read()
    else:
        #调整图片大小
        img = cv2.imread(path)
    img = cv2.resize(img,(widthImg,heightImg))
    imgContours=img.copy()
    imgFinal = img.copy()
    imgBiggestContours = img.copy()
    #转灰度
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #高斯滤波模糊
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    #边缘检测
    imgCanny = cv2.Canny(imgBlur,10,50)
    try:

        #找到轮廓
        contours,hierarchy=cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours,contours,-1,(0,255,0),10)
        #找到矩型
        rectCon=utlis.rectContour(contours)
        biggestContour = utlis.getCornerPoints(rectCon[0])
        gradePoints = utlis.getCornerPoints(rectCon[1])
        #print(biggestContour)
        print(img.shape)
        if biggestContour.size !=0 and gradePoints.size !=0:
            cv2.drawContours(imgBiggestContours,biggestContour,-1,(0,255,0),20)
            cv2.drawContours(imgBiggestContours,gradePoints,-1,(255,0,0),20)
            biggestContour=utlis.reorder(biggestContour)
            gradePoints=utlis.reorder(gradePoints)

            pt1 = np.float32(biggestContour)
            pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
            matrix = cv2.getPerspectiveTransform(pt1,pt2)
            imgWarpColored = cv2.warpPerspective(img,matrix,(widthImg,heightImg))

            ptG1 = np.float32(gradePoints)
            ptG2 = np.float32([[0,0],[325,0],[0,150],[325,150]])
            matrixG = cv2.getPerspectiveTransform(ptG1,ptG2)
            imgGradeDisplay = cv2.warpPerspective(img,matrixG,(325,150))
            #cv2.imshow("Grade",imgGradeDisplay)

            imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpGray,170,255,cv2.THRESH_BINARY_INV)[1]

            boxes=utlis.splitBoxes(imgThresh)
            #cv2.imshow("Test",boxes[2])
            #print(cv2.countNonZero(boxes[1]),cv2.countNonZero(boxes[2]))
            #判断每个框的白点数
            myPixelVal = np.zeros((questions,choices))
            countC = 0#行
            countR = 0#列
            for image in boxes:
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC] =totalPixels
                countC+=1
                if countC==choices:
                    countR+=1
                    countC=0
            #print(myPixelVal)
            #寻找最大数
            myIndex = []
            for x in range(0,questions):
                arr = myPixelVal[x]
                myIndexVal=np.where(arr==np.amax(arr))
                #print(myIndexVal[0])
                myIndex.append(myIndexVal[0][0])

            grading=[]
            for x in range(0,questions):
                if ans[x] == myIndex[x]:
                    grading.append(1)
                else:
                    grading.append(0)
            #print(grading)
            score = (sum(grading)/questions)*100
            #显示答案
            imgResult = imgWarpColored.copy()
            imgResult = utlis.showAnswers(imgResult, myIndex, grading,ans,questions, choices)
            imgRawDrawing = np.zeros_like(imgWarpColored)
            imgRawDrawing = utlis.showAnswers(imgRawDrawing,myIndex,grading,ans,questions,choices)
            invMatrix = cv2.getPerspectiveTransform(pt2,pt1)
            imgInWarp = cv2.warpPerspective(imgRawDrawing,invMatrix,(widthImg,heightImg))
            imgRawGrade = np.zeros_like(imgGradeDisplay)
            cv2.putText(imgRawGrade,str(int(score))+"%",(60,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3)
            invMatrixG = cv2.getPerspectiveTransform(pt2, pt1)
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade,invMatrixG,(widthImg,heightImg))

        imgFinal = cv2.addWeighted(imgFinal,1,imgInWarp,1.5,0)


        imgBlank = np.zeros_like(img)
        imageArray = ([img,imgGray,imgBlur,imgCanny],
                      [imgContours,imgBiggestContours,imgWarpColored,imgThresh],
                      [imgResult,imgRawDrawing,imgInWarp,imgFinal])
    except:
        imgBlank = np.zeros_like(img)
        imageArray = ([img, imgBlur, imgBlur, imgBlur],
                      [imgBlur, imgBlur, imgBlur, imgBlur],
                      [imgBlur, imgBlur, imgBlur, imgBlur])
    # lables = [["Original","Gray","Blur","Canny"],
    #           ["Contours","Biggest","Con","Warp","Threshold"],
    #           ["Result","Raw Drawing","Inv Warp","Final"]]
    #imgStacked = utlis.stackImages(imageArray,0.3,lables=lables)
    imgStacked = utlis.stackImages(imageArray, 0.3)
    cv2.imshow("Final",imgFinal)
    cv2.imshow("imgStacked",imgStacked)
    if cv2.waitKey(1) &0xFF ==27:
        break