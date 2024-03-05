import cv2
import ser
import numpy as np
import time
cap = cv2.VideoCapture(0)
ser.Connect()


def get_command(cap):
    try:
        ret,image = cap.read()
        # 转化为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 二值图像处理
        ret2, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        # 获得轮廓
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 拟合直线
        [vx, vy, x1, y1] = cv2.fitLine(contours[0], cv2.DIST_L2, 0, 0.01, 0.01)
        # 转化角度
        angle = np.float((np.arctan2(vy, vx) * 180 / np.pi))
        #将角度转换位指令
    except:
        command = 'a0'
    try:
        if angle < 0:
            command = 'c1'
        elif angle == 0:
            command = 'a1'
        elif angle > 0:
            command = 'b1'
    except:
        pass
    print(angle)
    print(command)
    return command

while True:
    #获得指令
    command = get_command(cap)
    #发送指令
    ser.SendCommand(command)
    time.sleep(0.5)
cap.release()
cv2.destroyAllWindows()