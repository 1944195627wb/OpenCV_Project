import numpy as np
import cv2


def get_command(cap):
    ret1, image = cap.read()
    # 转化为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值图像处理
    ret2, binary = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
    try:
        # 获得轮廓
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 拟合直线
        [vx, vy, x1, y1] = cv2.fitLine(contours[0], cv2.DIST_L2, 0, 0.01, 0.01)
        # 转化角度
        angle = np.float(np.arctan2(vy, vx) * 180 / np.pi)
    except IndexError:
        angle = 0.0
    #将角度转换位指令
    if angle < 0:
        angle = (90-np.abs(angle))*3/2
        if angle >= 60:
            command = 'c3'
        elif angle >= 30:
            command = 'c2'
        else:
            command = 'c1'
    elif angle == 0:
        command = 'a0'
    elif angle > 0:
        if angle == 90:
            command = 'a1'
        else:
            angle = (90-angle)*3/2
            if angle >= 60:
                command = 'b3'
            elif angle >= 30:
                command = 'b2'
            else:
                command = 'b1'
    return command






        



