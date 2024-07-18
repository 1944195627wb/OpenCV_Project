import cv2
import numpy as np
import utlis
import pytesseract

class DocumentScanner:
    def __init__(self, web_cam_feed=False, image_path="3.jpg"):
        self.web_cam_feed = web_cam_feed
        self.image_path = image_path
        self.cap = cv2.VideoCapture(0) if web_cam_feed else None
        self.count = 0
        self.height_img = 640
        self.width_img = 480

    def initialize_trackbars(self):
        utlis.initializeTrackbars()

    def val_trackbars(self):
        return utlis.valTrackbars()

    def process_image(self, img):
        img = cv2.resize(img, (self.width_img, self.height_img))
        img_blank = np.zeros((self.height_img, self.width_img, 3), np.uint8)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
        thres = self.val_trackbars()
        img_threshold = cv2.Canny(img_blur, thres[0], thres[1])
        kernel = np.ones((5, 5))
        img_dial = cv2.dilate(img_threshold, kernel, iterations=2)
        img_threshold = cv2.erode(img_dial, kernel, iterations=1)
        return img, img_gray, img_threshold

    def find_document(self, img, img_threshold):
        img_contours = img.copy()
        img_big_contour = img.copy()
        contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 10)
        biggest, max_area = utlis.biggestContour(contours)

        return biggest, img_big_contour

    def process_document(self, biggest, img, img_gray):
        if biggest.size != 0:
            biggest = utlis.reorder(biggest)
            cv2.drawContours(img, biggest, -1, (0, 255, 0), 20)
            img = utlis.drawRectangle(img, biggest, 2)
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0, 0], [self.width_img, 0], [0, self.height_img], [self.width_img, self.height_img]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            img_warp_colored = cv2.warpPerspective(img, matrix, (self.width_img, self.height_img))
            img_warp_colored = img_warp_colored[20:img_warp_colored.shape[0] - 20, 20:img_warp_colored.shape[1] - 20]
            img_warp_colored = cv2.resize(img_warp_colored, (self.width_img, self.height_img))
            return img_warp_colored
        else:
            return None

    def scan_document(self, img_warp_colored):
        img_warp_gray = cv2.cvtColor(img_warp_colored, cv2.COLOR_BGR2GRAY)
        img_adaptive_thre = cv2.adaptiveThreshold(img_warp_gray, 255, 1, 1, 7, 2)
        img_adaptive_thre = cv2.bitwise_not(img_adaptive_thre)
        img_adaptive_thre = cv2.medianBlur(img_adaptive_thre, 3)
        boxes = pytesseract.image_to_data(img_adaptive_thre, lang='chi_sim')
        h_img, w_img = img_adaptive_thre.shape
        for x, b in enumerate(boxes.splitlines()):
            if x != 0 and len(b) == 12:
                x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                cv2.rectangle(img_adaptive_thre, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.putText(img_adaptive_thre, b[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (55, 55, 255), 2)
        return img_adaptive_thre

    def run(self):
        self.initialize_trackbars()
        while True:
            if self.web_cam_feed:
                ret, img = self.cap.read()
            else:
                img = cv2.imread(self.image_path)
            img, img_gray, img_threshold = self.process_image(img)
            biggest, img_big_contour = self.find_document(img, img_threshold)
            if biggest is not None:
                img_warp_colored = self.process_document(biggest, img, img_gray)
                if img_warp_colored is not None:
                    img_adaptive_thre = self.scan_document(img_warp_colored)
                    cv2.imshow("Result", img_adaptive_thre)
            else:
                cv2.imshow("Result", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if self.web_cam_feed:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    scanner = DocumentScanner(web_cam_feed=False, image_path="3.jpg")
    scanner.run()
