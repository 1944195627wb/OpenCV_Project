# from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton,QPlainTextEdit
#
# app = QApplication([])
# window = QMainWindow()
# window.resize(600,480)
# window.move(300,300)
# window.setWindowTitle('文件扫描与文字提取系统')
import cv2
import tkinter as tk
from tkinter import filedialog
from DocumentScanner import DocumentScanner

class DocumentScannerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("文件扫描与文字提取系统")

        # 变量
        self.web_cam_feed = False
        self.image_path = ""

        # 主框架
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)

        # 模式选择框架
        self.mode_frame = tk.LabelFrame(self.main_frame, text="模式选择")
        self.mode_frame.pack(fill=tk.X, padx=10, pady=5)

        self.image_mode = tk.Radiobutton(self.mode_frame, text="图像", command=self.select_image_mode)
        self.image_mode.pack(side=tk.LEFT, padx=10)

        self.video_mode = tk.Radiobutton(self.mode_frame, text="视频", command=self.select_video_mode)
        self.video_mode.pack(side=tk.LEFT, padx=10)

        self.webcam_mode = tk.Radiobutton(self.mode_frame, text="摄像头", command=self.select_webcam_mode)
        self.webcam_mode.pack(side=tk.LEFT, padx=10)

        # 图像路径输入
        self.image_path_frame = tk.Frame(self.main_frame)
        self.image_path_frame.pack(fill=tk.X, padx=10, pady=5)

        self.image_path_label = tk.Label(self.image_path_frame, text="图像路径:")
        self.image_path_label.pack(side=tk.LEFT)

        self.image_path_entry = tk.Entry(self.image_path_frame, width=40)
        self.image_path_entry.pack(side=tk.LEFT, padx=(5, 0))

        self.browse_button = tk.Button(self.image_path_frame, text="浏览", command=self.browse_image)
        self.browse_button.pack(side=tk.LEFT, padx=(5, 0))

        # 开始按钮
        self.start_button = tk.Button(self.main_frame, text="开始扫描", command=self.start_scan)
        self.start_button.pack(fill=tk.X, padx=10, pady=5)

        # OpenCV 变量
        self.cap = None

    def select_image_mode(self):
        self.web_cam_feed = False

    def select_video_mode(self):
        self.web_cam_feed = False
        self.image_path_entry.delete(0, tk.END)

    def select_webcam_mode(self):
        self.web_cam_feed = True
        self.image_path_entry.delete(0, tk.END)

    def browse_image(self):
        self.image_path = filedialog.askopenfilename()
        self.image_path_entry.delete(0, tk.END)
        self.image_path_entry.insert(0, self.image_path)

    def start_scan(self):
        self.image_path = self.image_path_entry.get()
        scanner = DocumentScanner(self.web_cam_feed, self.image_path)
        scanner.run()


def main():
    root = tk.Tk()
    app = DocumentScannerUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
