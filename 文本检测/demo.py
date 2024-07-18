import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'D:\\APPs\\Tesseract\\tesseract.exe'
img = cv2.imread('2.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#print(pytesseract.image_to_string(img,lang='chi_sim'))

# #检测字符
# boxes = pytesseract.image_to_boxes(img,lang='chi_sim')
# hImg,wImg,_ = img.shape
# for b in boxes.splitlines():
#     print(b)
#     b = b.split(' ')
#     print(b)
#     x,y,w,h=int(b[1]),int(b[2]),int(b[3]),int(b[4])
#     cv2.rectangle(img,(x,hImg-y),(w,hImg-h),(0,0,255),1)
#     cv2.putText(img,b[0],(x,hImg-y+25),cv2.FONT_HERSHEY_COMPLEX,1,(55,55,255),2)

#检测单词
boxes = pytesseract.image_to_data(img,lang='chi_sim')
hImg,wImg,_ = img.shape
for x,b in enumerate(boxes.splitlines()):
    if x!=0:
        b = b.split()
        print(b)
        if len(b)==12:
            x,y,w,h=int(b[6]),int(b[7]),int(b[8]),int(b[9])
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
            cv2.putText(img,b[11],(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(55,55,255),2)

# #检测单词,配置只能检测数字
# hImg,wImg,_ = img.shape
# cong=r'--oem 3 --psm 6 outputbase digits'
# boxes = pytesseract.image_to_data(img,lang='chi_sim',config=cong)
# for x,b in enumerate(boxes.splitlines()):
#     if x!=0:
#         b = b.split()
#         print(b)
#         if len(b)==12:
#             x,y,w,h=int(b[6]),int(b[7]),int(b[8]),int(b[9])
#             cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
#             cv2.putText(img,b[11],(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(55,55,255),2)

cv2.imshow("Result",img)
cv2.waitKey(0)
