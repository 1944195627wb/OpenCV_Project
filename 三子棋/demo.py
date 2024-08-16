import cv2
import numpy as np
import utlis

#图片和其宽高
path='chess/test/3.jpg'
widthImg = 640
heightImg = 480
questions = 3
choices = 3



#是否使用摄像头
webcamFeed = False


#摄像头设置
cap = cv2.VideoCapture(0)
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
    imgCanny = cv2.Canny(imgBlur,0,50)
    #cv2.imshow("Canny",imgCanny)

    # try:

    #找到轮廓
    contours,hierarchy=cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    maxArea = 0
    maxRect = None
    maxContour = None
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        if abs(rect[1][0]-rect[1][1])<30 and rect[1][0]>100 and rect[1][1]>100:
            rectArea = rect[1][0]*rect[1][1]
            if rectArea>maxArea:
                maxArea = rectArea
                maxRect = rect
                maxContour = contour

    x,y,w,h = cv2.boundingRect(maxContour)
    cv2.rectangle(imgContours,(x,y),(x+w,y+h),(0,255,0),2)
    box = cv2.boxPoints(maxRect)
    box = np.intp(box)
    #print(box)
    cv2.drawContours(imgContours,[box],0,(255,255,0),2)

    #cv2.imshow("Contours",imgContours)

    
    

    #rectCon = utlis.rectContour(contours)
    #biggestContour = utlis.getCornerPoints()
    #gradePoints = utlis.getCornerPoints()


    
    
    if  maxRect!=None:
        cv2.rectangle(imgBiggestContours,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.drawContours(imgBiggestContours,[box],0,(255,255,0),2)
        
        newRect = np.array([box[1],box[0],box[2],box[3]])
        pt1 = np.float32(newRect)
        pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
        matrix = cv2.getPerspectiveTransform(pt1,pt2)
        imgWarpColored = cv2.warpPerspective(img,matrix,(widthImg,heightImg))

        

        imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        imgThresh = cv2.threshold(imgWarpGray,190,255,cv2.THRESH_BINARY)[1]
        imgThreshInv = cv2.threshold(imgWarpGray,50, 255, cv2.THRESH_BINARY_INV)[1]

        # cv2.imshow("Thresh",imgThresh)
        # cv2.imshow("ThreshInv",imgThreshInv)
        
        boxes=utlis.splitBoxes(imgThresh,imgThreshInv,questions,choices)
        

        
        
        


        

        
        # 计算每个小块中的白色和黑色像素数
        myPixelVal = np.zeros((questions, choices))
        myBlackPixelVal = np.zeros((questions, choices))
        countC = 0  # 列计数
        countR = 0  # 行计数
        
        for image in (boxes):
            countC += 1
            contours, hierarchy = cv2.findContours(image[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours_inv, hierarchy_inv = cv2.findContours(image[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            maxcircleArea = 0
            maxcircle = None
            maxcircleContour = None
            flag = None
            
            

            for contour in contours:
                if len(contour) > 5:
                    ellipse = cv2.fitEllipse(contour)
                    circleArea = ellipse[1][0] * ellipse[1][1]
                    
                    if 10000<circleArea<30000:
                        if circleArea > maxcircleArea:
                            maxcircleArea = circleArea
                            maxcircle = ellipse
                            maxcircleContour = contour
                            flag = 'white'

            for contour in contours_inv:
                if len(contour) > 5:
                    ellipse = cv2.fitEllipse(contour)
                    circleArea = ellipse[1][0] * ellipse[1][1]
                    if 10000<circleArea<30000:
                        if circleArea > maxcircleArea:
                            maxcircleArea = circleArea
                            maxcircle = ellipse
                            maxcircleContour = contour
                            flag = 'black'
            
            
            
            if maxcircle != None:
                if flag == 'white':
                    #计算椭圆内的白色像素数
                    mask = np.zeros_like(image[0])
                    cv2.ellipse(mask, maxcircle, (255, 255, 255), -1)
                    whitePixel = cv2.countNonZero(cv2.bitwise_and(image[0], mask))
                    myPixelVal[countR][countC-1] = whitePixel
                    #计算椭圆内的黑色像素数
                    mask = np.zeros_like(image[1])
                    cv2.ellipse(mask, maxcircle, (255, 255, 255), -1)
                    blackPixel = cv2.countNonZero(cv2.bitwise_and(image[1], mask))
                    myBlackPixelVal[countR][countC-1] = blackPixel

                elif flag == 'black':
                    #计算椭圆内的白色像素数
                    mask = np.zeros_like(image[0])
                    cv2.ellipse(mask, maxcircle, (255, 255, 255), -1)
                    whitePixel = cv2.countNonZero(cv2.bitwise_and(image[0], mask))
                    myPixelVal[countR][countC-1] = whitePixel
                    #计算椭圆内的黑色像素数
                    mask = np.zeros_like(image[1])
                    cv2.ellipse(mask, maxcircle, (255, 255, 255), -1)
                    blackPixel = cv2.countNonZero(cv2.bitwise_and(image[1], mask))
                    myBlackPixelVal[countR][countC-1] = blackPixel
                    
            else:
                myPixelVal[countR][countC-1] = 0
                myBlackPixelVal[countR][countC-1] = 0


            

                    
                



                    
                

            


            if countC == choices:
                countR += 1
                countC = 0

        

        imgResult = imgWarpColored.copy()

        boxWidth = imgWarpColored.shape[1] // choices
        boxHeight = imgWarpColored.shape[0] // questions
        


        
        
        
        imgResult = imgWarpColored.copy()
        imgRawDrawing = np.zeros_like(imgWarpColored)

        for r in range(questions):
            for c in range(choices):
                
                if myPixelVal[r][c]==0 and myBlackPixelVal[r][c]==0:
                    cv2.putText(imgResult, "None", (c * boxWidth + boxWidth // 2 - 10, r * boxHeight + boxHeight // 2 + 10),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
                    cv2.putText(imgRawDrawing, "None", (c * boxWidth + boxWidth // 2 - 10, r * boxHeight + boxHeight // 2 + 10),
                                cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255), 2)
                
                elif myPixelVal[r][c]>myBlackPixelVal[r][c]:
                    cv2.putText(imgResult, "White", (c * boxWidth + boxWidth // 2 - 10, r * boxHeight + boxHeight // 2 + 10),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
                    cv2.putText(imgRawDrawing, "White", (c * boxWidth + boxWidth // 2 - 10, r * boxHeight + boxHeight // 2 + 10),
                                cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255), 2)
                    
                elif myPixelVal[r][c]<myBlackPixelVal[r][c]:
                    
                    cv2.putText(imgResult, "Black", (c * boxWidth + boxWidth // 2 - 10, r * boxHeight + boxHeight // 2 + 10),
                                cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255), 2)
                    cv2.putText(imgRawDrawing, "Black", (c * boxWidth + boxWidth // 2 - 10, r * boxHeight + boxHeight // 2 + 10),
                                cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255), 2)

                    
        
        
        invMatrix = cv2.getPerspectiveTransform(pt2,pt1)
        imgInWarp = cv2.warpPerspective(imgRawDrawing,invMatrix,(widthImg,heightImg))
        invMatrixG = cv2.getPerspectiveTransform(pt2, pt1)

    imgFinal = cv2.addWeighted(imgFinal,1,imgInWarp,1.5,0)
    


    imgBlank = np.zeros_like(img)
    imageArray = ([img,imgGray,imgBlur,imgCanny],
                [imgContours,imgBiggestContours,imgWarpColored,imgThresh],
                [imgResult,imgRawDrawing,imgInWarp,imgFinal])
    # except:
    #     imgBlank = np.zeros_like(img)
    #     imageArray = ([img, imgBlur, imgBlur, imgBlur],
    #                   [imgBlur, imgBlur, imgBlur, imgBlur],
    #                   [imgBlur, imgBlur, imgBlur, imgBlur])

    #lables = [["Original","Gray","Blur","Canny"],
            #   ["Contours","Biggest","Con","Warp","Threshold"],
            #   ["Result","Raw Drawing","Inv Warp","Final"]]
    #imgStacked = utlis.stackImages(imageArray,0.3,lables=lables)
    imgStacked = utlis.stackImages(imageArray, 0.3)
    cv2.imshow("Final",imgFinal)
    cv2.imshow("imgStacked",imgStacked)
    cv2.imshow('result', imgResult)
    if cv2.waitKey(1) &0xFF ==27:
        break