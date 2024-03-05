#import ser
import cv2
import numpy as np
cap = cv2.VideoCapture(0)
#ser.Connect()
global x, angle, x_change
angle = 0
while True:
    try:
        #ret1,image = cap.read()
        image = cv2.imread("D:/photo/youwan.png")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #二值图像处理
        ret2, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
        cv2.imshow("binary", binary)
        cv2.imshow("image", image)
        cv2.waitKey()
        contours,hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center_x= image.shape[1]//2
        center_y = image.shape[0]//2
        max_area=0
        max_con = None
        for con in contours:
            area = cv2.contourArea(con)
            if area>max_area:
                max_area = area
                max_con = con
        if max_con is not None:
            M =cv2.moments(max_con)
            if M['m00'] != 0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
                x_change = x - center_x
                angle = x_change/5
            if angle >=0:
                if angle<3.2:
                    command ='a1'
                else:
                    command = 'c1'
            else:
                if angle>=-3.2:
                    command = 'a1'
                else:
                    command = 'b1'
    except:
        command = 'a1'
    print(angle)
    print(command)
    #ser.SendCommand(command)
cap.release()
cv2.destroyAllWindows()