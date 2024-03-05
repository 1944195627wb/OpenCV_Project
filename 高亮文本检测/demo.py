from utlis import *
import pytesseract

path = 'test.png'
hsv = [0, 65, 59, 255, 0, 255]
pytesseract.pytesseract.tesseract_cmd = 'D:\\APPs\\Tesseract\\tesseract.exe'

img = cv2.imread(path)
# cv2.imshow("Original",img)

imgResult = detectColor(img, hsv)

imgContours, contours = getContours(imgResult, img, showCanny=True,
                                    minArea=1000, filter=4,
                                    cThr=[100, 150], draw=True)
cv2.imshow("imgContours",imgContours)
print(len(contours))

roiList = getRoi(img, contours)
# cv2.imshow("TestCrop",roiList[2])
roiDisplay(roiList)

highlightedText = []
for x, roi in enumerate(roiList):
    # print(pytesseract.image_to_string(roi))
    print(pytesseract.image_to_string(roi))
    highlightedText.append(pytesseract.image_to_string(roi))
    if cv2.waitKey(1) & 0xFF == 27:
        break

saveText(highlightedText)

imgStack = stackImages(0.6, ([img, imgResult, imgContours]))
cv2.imshow("Stacked Images", imgStack)
cv2.waitKey(0)

