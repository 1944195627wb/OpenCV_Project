import cv2
import numpy as np


def circle_angle(dilation):
    angle = []
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    [vx, vy, x1, y1] = cv2.fitLine(contours[0], cv2.DIST_L2, 0, 0.01, 0.01)
    angle.append(np.float(np.arctan2(vy, vx) * 180 / np.pi))
    x2 = x1 - 200 * vx
    y2 = y1 - 200 * vy
    cv2.line(dilation, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return angle
