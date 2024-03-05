import numpy as np
import cv2
import line_angle
import circle_angle

def circle_angle(dilation):
    angle = []
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    [vx, vy, x1, y1] = cv2.fitLine(contours[0], cv2.DIST_L2, 0, 0.01, 0.01)
    angle.append(np.float(np.arctan2(vy, vx) * 180 / np.pi))
    x2 = x1 - 200 * vx
    y2 = y1 - 200 * vy
    cv2.line(dilation, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return angle


def get_command(cap):
    ret1, image = cap.read()
    cv2.imshow('image', image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值图像处理
    ret2, binary = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY_INV)
    # 膨胀
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(binary, kernel, iterations=2)
    try:
        # 边缘检测
        contours = cv2.Canny(binary, 150, 200)
        angle = line_angle.line_angle(contours)
    except TypeError:
        angle = circle_angle.circle_angle(dilation)
    #将角度转化为指令
    try:
        if angle[0] <= 0:
            angle = np.abs(angle[0])
            if angle >= 60:
                command = 'c1'
            elif angle >= 30:
                command = 'c2'
            else:
                command = 'c3'
        else:
            angle = angle[0]
            if 60 <= angle < 90:
                command = 'b1'
            elif 30 <= angle < 60:
                command = 'b2'
            else:
                command = 'b3'
    except IndexError:
        command ='a3'
    return command