import os
import cv2
import torch
import numpy as np

from PySide6.QtGui import QIcon, QPixmap
from PySide6 import QtWidgets, QtCore, QtGui
from ultralytics import YOLO


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_gui()
        self.model = None
        self.timer = QtCore.QTimer()
        self.timer1 = QtCore.QTimer()
        self.cap = None
        self.video = None
        self.timer.timeout.connect(self.camera_show)
        self.timer1.timeout.connect(self.video_show)

        self.background_image = QPixmap(r"C:\Users\57704\Desktop\tri_pre\11.png")

    def init_gui(self):
        self.setFixedSize(960, 440)
        self.setWindowTitle('Pikachu!')
        self.setWindowIcon(QIcon("🅱️ "))

        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)

        mainLayout = QtWidgets.QVBoxLayout(centralWidget)

        topLayout = QtWidgets.QHBoxLayout()

        self.oriScrollArea = QtWidgets.QScrollArea(self)
        self.oriScrollArea.setWidgetResizable(True)
        self.oriVideoLabel = QtWidgets.QLabel(self)
        self.oriScrollArea.setWidget(self.oriVideoLabel)

        self.detectScrollArea = QtWidgets.QScrollArea(self)
        self.detectScrollArea.setWidgetResizable(True)
        self.detectlabel = QtWidgets.QLabel(self)
        self.detectScrollArea.setWidget(self.detectlabel)

        topLayout.addWidget(self.oriScrollArea)
        topLayout.addWidget(self.detectScrollArea)

        mainLayout.addLayout(topLayout)

        groupBox = QtWidgets.QGroupBox(self)
        bottomLayout = QtWidgets.QVBoxLayout(groupBox)
        mainLayout.addWidget(groupBox)

        btnLayout = QtWidgets.QHBoxLayout()
        self.selectModel = QtWidgets.QPushButton('📂选择模型')
        self.selectModel.setFixedSize(100, 50)
        self.selectModel.clicked.connect(self.load_model)
        
        self.openVideoBtn = QtWidgets.QPushButton('🎞️视频文件')
        self.openVideoBtn.setFixedSize(100, 50)
        self.openVideoBtn.clicked.connect(self.start_video)
        self.openVideoBtn.setEnabled(False)
        
        self.openImageBtn = QtWidgets.QPushButton('🖼️图像文件')
        self.openImageBtn.setFixedSize(100, 50)
        self.openImageBtn.clicked.connect(self.start_image)
        
        self.openCamBtn = QtWidgets.QPushButton('📹摄像头')
        self.openCamBtn.setFixedSize(100, 50)
        self.openCamBtn.clicked.connect(self.start_camera)
        
        self.stopDetectBtn = QtWidgets.QPushButton('🛑停止')
        self.stopDetectBtn.setFixedSize(100, 50)
        self.stopDetectBtn.setEnabled(False)
        self.stopDetectBtn.clicked.connect(self.stop_detect)
        
        self.exitBtn = QtWidgets.QPushButton('⏹退出')
        self.exitBtn.setFixedSize(100, 50)
        self.exitBtn.clicked.connect(self.close)

        btnLayout.addWidget(self.selectModel)
        btnLayout.addWidget(self.openVideoBtn)
        btnLayout.addWidget(self.openImageBtn)
        btnLayout.addWidget(self.openCamBtn)
        btnLayout.addWidget(self.stopDetectBtn)
        btnLayout.addWidget(self.exitBtn)
        
        bottomLayout.addLayout(btnLayout)

    def paintEvent(self, event):
        
        painter = QtGui.QPainter(self)
        painter.drawPixmap(self.rect(), self.background_image)
        super().paintEvent(event)

    def start_image(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选取图像文件", filter='*.png;*.jpg;*.jpeg;*.bmp')
        if fileName:
            image = cv2.imread(fileName)
            if image is not None:
                
                h, w, _ = image.shape
                
                
                image_resized = cv2.resize(image, (448, 352))
                
                results = self.model(image_resized, imgsz=[448, 352], device='cuda') if torch.cuda.is_available() else self.model(image_resized, imgsz=[448, 352], device='cpu')
                
                detected_image = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)
                detected_image = QtGui.QImage(detected_image.data, detected_image.shape[1], detected_image.shape[0], QtGui.QImage.Format_RGB888)
                self.detectlabel.setPixmap(QtGui.QPixmap.fromImage(detected_image))
                self.detectlabel.setScaledContents(True)

                
                original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
                original_image = cv2.resize(original_image, (448, 352))
                original_image = QtGui.QImage(original_image.data, original_image.shape[1], original_image.shape[0], QtGui.QImage.Format_RGB888)
                self.oriVideoLabel.setPixmap(QtGui.QPixmap.fromImage(original_image))
                self.oriVideoLabel.setScaledContents(True)  

        
                self.oriVideoLabel.setMinimumSize(300, 300) 
                self.oriScrollArea.setMinimumSize(300, 300)   

    

    def start_camera(self):
        self.timer1.stop()
        if self.cap is None:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if self.cap.isOpened():
            self.timer.start(50)
        self.stopDetectBtn.setEnabled(True)

    def camera_show(self):
        ret, frame = self.cap.read()
        if ret:
            if self.model is not None:
                frame = cv2.resize(frame, (448, 352))
                frame1 = self.model(frame, imgsz=[448, 352], device='cuda') if torch.cuda.is_available() \
                    else self.model(frame, imgsz=[448, 352], device='cpu')
                frame1 = cv2.cvtColor(frame1[0].plot(), cv2.COLOR_RGB2BGR)
                frame1 = QtGui.QImage(frame1.data, frame1.shape[1], frame1.shape[0], QtGui.QImage.Format_RGB888)
                self.detectlabel.setPixmap(QtGui.QPixmap.fromImage(frame1))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            self.oriVideoLabel.setPixmap(QtGui.QPixmap.fromImage(frame))
            self.oriVideoLabel.setScaledContents(True)

    def start_video(self):
        if self.timer.isActive():
            self.timer.stop()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选取视频文件", filter='*.mp4')
        if os.path.isfile(fileName):
            self.video = cv2.VideoCapture(fileName)
            fps = self.video.get(cv2.CAP_PROP_FPS)
            self.timer1.start(int(1000/fps))
        else:
            print("Reselect video")

    def video_show(self):
        ret, frame = self.video.read()
        if ret:
            if self.model is not None:
                frame = cv2.resize(frame, (448, 352))
                frame1 = self.model(frame, imgsz=[448, 352], device='cuda') if torch.cuda.is_available() \
                    else self.model(frame, imgsz=[448, 352], device='cpu')
                frame1 = cv2.cvtColor(frame1[0].plot(), cv2.COLOR_RGB2BGR)
                frame1 = QtGui.QImage(frame1.data, frame1.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
                self.detectlabel.setPixmap(QtGui.QPixmap.fromImage(frame1))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            self.oriVideoLabel.setPixmap(QtGui.QPixmap.fromImage(frame))
            self.oriVideoLabel.setScaledContents(True)
        else:
            self.timer1.stop()
            img = cv2.cvtColor(np.zeros((500, 500), np.uint8), cv2.COLOR_BGR2RGB)
            img = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
            self.oriVideoLabel.setPixmap(QtGui.QPixmap.fromImage(img))
            self.detectlabel.setPixmap(QtGui.QPixmap.fromImage(img))
            self.video.release()
            self.video = None

    def load_model(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选取模型权重", filter='*.pt')
        if fileName.endswith('.pt'):
            self.model = YOLO(fileName)
        else:
            print("Reselect model")

        self.openVideoBtn.setEnabled(True)
        self.stopDetectBtn.setEnabled(True)

    def stop_detect(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.timer1.isActive():
            self.timer1.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video = None
        # img = cv2.cvtColor(np.zeros((500, 500), np.uint8), cv2.COLOR_BGR2RGB)
        # img = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        # self.oriVideoLabel.setPixmap(QtGui.QPixmap.fromImage(img))
        # self.detectlabel.setPixmap(QtGui.QPixmap.fromImage(img))
        self.oriVideoLabel.clear()
        self.detectlabel.clear()
    
        self.update()

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.timer.isActive():
            self.timer.stop()
        exit()


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MyWindow()
    window.show()
    app.exec()