import cv2
import numpy as np


def line_angle(contours):
    lines = cv2.HoughLinesP(contours, 1, np.pi / 180,60, minLineLength=275, maxLineGap=50)
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






