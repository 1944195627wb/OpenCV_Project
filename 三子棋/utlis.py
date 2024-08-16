import numpy as np
import cv2


def stackImages(imgArray, scale, lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        # print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(lables[d]) * 13 + 27, 30 + eachImgHeight * d), (255, 255, 255),
                              cv2.FILLED)
                cv2.putText(ver, lables[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20), cv2.FONT_HERSHEY_COMPLEX,
                            0.7, (255, 0, 255), 2)
    return ver

def rectContour(contours):
    rectCon = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                
                if cv2.isContourConvex(approx):
                    
                    angles = []
                    for i in range(4):
                        p1 = approx[i][0]
                        p2 = approx[(i+1) % 4][0]
                        p3 = approx[(i+2) % 4][0]

                        v1 = p2 - p1
                        v2 = p3 - p2

                        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                        angles.append(np.degrees(angle))

                    
                    if all(70 <= angle <= 110 for angle in angles):
                        rectCon.append(contour)
    
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)
    return rectCon

def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
    return approx

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    #给出展平后最小值的下标
    myPointsNew[0] = myPoints[np.argmin(add)]#[0,0]
    myPointsNew[3] = myPoints[np.argmax(add)]#[w,h]
    diff = np.diff(myPoints,axis=1)
    #x最大y最小
    myPointsNew[1] = myPoints[np.argmin(diff)]#[w,0]
    #x最小y最大
    myPointsNew[2] = myPoints[np.argmax(diff)]#[h,0]
    return myPointsNew

def resize_and_pad(img, grid_size):
    h, w = img.shape[:2]
    new_h = (h // grid_size + 1) * grid_size
    new_w = (w // grid_size + 1) * grid_size
    padded_img = cv2.copyMakeBorder(img, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded_img

def splitBoxes(img_thresh, img_thresh_inv, rows=3, cols=3):
    
    img_thresh = resize_and_pad(img_thresh, max(rows, cols))
    img_thresh_inv = resize_and_pad(img_thresh_inv, max(rows, cols))
    
    
    split_rows_thresh = np.vsplit(img_thresh, rows)
    split_rows_thresh_inv = np.vsplit(img_thresh_inv, rows)
    
    
    boxes = []
    for r in range(rows):
        split_cols_thresh = np.hsplit(split_rows_thresh[r], cols)
        split_cols_thresh_inv = np.hsplit(split_rows_thresh_inv[r], cols)
        
        for c in range(cols):
            boxes.append([split_cols_thresh[c], split_cols_thresh_inv[c]])
    
    return boxes

def showAnswers(img,myIndex,grading,ans,questions,choices):
    secW = int(img.shape[1]/choices)
    secH = int(img.shape[0]/questions)

    for x in range(0,questions):
        myAns = myIndex[x]
        cX = int((myAns*secW)+secW/2)
        cY = int((x*secH)+ secH/2)
        if grading[x] == 1:
            myColor = (0,255,0)
        else:
            myColor = (0,0,255)
            correctAns = ans[x]
            correctX = int((correctAns*secW)+secW/2)
            correctY = int((x * secH) + secW / 2)
            cv2.circle(img,(correctX,correctY),50,(0,255,0),cv2.FILLED)

        cv2.circle(img,(cX,cY),50,myColor,cv2.FILLED)
    return img