import cv2
import numpy as np
import pickle

polygons = []  # all the polygons and their points
path = []  # current single polygon

img = cv2.imread('imgBoard.png')


def mousePoints(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        path.append([x, y])


while True:

    for point in path:
        cv2.circle(img, point, 7, (0, 0, 255), cv2.FILLED)

    pts = np.array(path, np.int32).reshape((-1, 1, 2))
    img = cv2.polylines(img, [pts], True, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mousePoints)
    key = cv2.waitKey(1)

    if key == ord('q'):
        score = int(input("Enter Score: "))
        polygons.append([path, score])
        print("Total Polygons: ", len(polygons))
        path = []
    if key == ord("p"):
        with open('polygons','wb') as f:
            print(polygons)
            pickle.dump(polygons,f)
        break
