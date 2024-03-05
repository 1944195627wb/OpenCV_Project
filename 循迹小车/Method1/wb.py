import ser
import time
import cv2
import numpy as np
cap = cv2.VideoCapture(0)
ser.Connect()


def circle_angle(dilation):
    angle = []
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    [vx, vy, x1, y1] = cv2.fitLine(contours[0], cv2.DIST_L2, 0, 0.01, 0.01)
    angle.append(np.float(np.arctan2(vy, vx) * 180 / np.pi))
    return angle


def get_command(cap):
    ret,image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值图像处理
    ret2, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    try:
        # 边缘检测
        contours = cv2.Canny(binary,150,200)
        angle = line_angle(contours)
    except TypeError:
        angle = circle_angle(binary)
    #将角度转化为指令
    print(angle)
    try:
        if angle[0] ==180:
            command='b1'
        elif angle[0] ==0:
            command ='c1'
        elif angle[0] <0:
            command = 'c1'
        else:
            command = 'b1'
    except:
        command ='a0'

    print(command)
    return command


def line_angle(contours):
    lines = cv2.HoughLinesP(contours, 1, np.pi / 180,60, minLineLength=150, maxLineGap=50)
    # 去除重复的直线
    unique_lines = []
    angle = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if len(unique_lines) == 0:
            unique_lines.append(line)
        else:
            _ = True
            for _line in unique_lines:
                ux1, uy1, ux2, uy2 = _line[0]
                dist = np.abs((y2 - y1) * ux1 - (x2 - x1) * uy1 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                if dist < 150:  # 设定距离阈值为85个像素
                    _ = False
                    break
            if _:
                unique_lines.append(line)
    for line in unique_lines:
        x1,y1,x2,y2 = line[0]
        if x1 == x2:
            pass
        elif y1 == y2:
            if x2 > contours.shape[1]*7/10:
                angle.append(0)
            else:
                angle.append(180)
        else:
            angle.append(np.arctan2(y2-y1,x2-x1)*180/np.pi)
    return angle


while True:
    #获得指令
    command =get_command(cap)
    #发出指令
    ser.SendCommand(command)
cap.release()
cv2.destroyAllWindows()