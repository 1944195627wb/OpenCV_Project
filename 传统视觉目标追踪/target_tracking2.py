#导入库
import cv2
#读取视频
video = cv2.VideoCapture("D:/video/csgo.mp4")
#初始化原始坐标
x1,y1,x2,y2=0,0,0,0

#定义鼠标选中函数
def select_object(event,x,y,flags,param):
    #将函数中的变量位为全局变量
    global x1,y1,x2,y2
    #如果事件为鼠标按下
    if event==cv2.EVENT_LBUTTONDOWN:
        #左上坐标为此时鼠标的位置
        x1,y1 = x,y
        x2,y2 = x,y
    #如果事件为鼠标松开
    elif event == cv2.EVENT_LBUTTONUP:
        #右下坐标为此时鼠标位置
        x2,y2 = x,y
    #避免从右下到左上的情况
    x1,x2 = min(x1,x2),max(x1,x2)
    y1,y2 = min(y1,y2),max(y1,y2)
    #画出选中区域
    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2)
    #显示选中区域
    cv2.imshow("SelectObject",frame)

#读取第一帧
ret1,frame = video.read()
#创建窗口
cv2.namedWindow('SelectObject')
#作用鼠标回调函数
cv2.setMouseCallback("SelectObject",select_object)
#停止直至按下任意键结束
cv2.waitKey()
while True:
    #避免视频结束报错
    try:
        #读取视频流
        ret,image = video.read()
        #截取框选部分
        image1 = frame[y1:y2,x1:x2]
        #获得形状大小
        th,tw = image1.shape[:2]
        #模块匹配
        rv = cv2.matchTemplate(image,image1,cv2.TM_SQDIFF)
        #获得最小，从而得到匹配最大的区域
        minVal,maxVal,minLoc,maxLoc = cv2.minMaxLoc(rv)
        topLeft = minLoc
        #获得右下角坐标
        bottomRight=(topLeft[0]+tw,topLeft[1]+th)
        #画出框选部分
        cv2.rectangle(image,topLeft,bottomRight,(255,255,0),2)
        #显示
        cv2.imshow("video",image)
        #如果按下空格键则结束进程
        if cv2.waitKey(20) == ord(' '):
            break
    except:
        pass
#关闭视频流
video.release()
#关闭窗口
cv2.destroyAllWindows()