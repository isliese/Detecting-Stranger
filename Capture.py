import sys
import cv2
import os
import datetime
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton, QInputDialog
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

class CameraWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Camera')
        self.setGeometry(50, 50, 320, 240)

        self.layout = QVBoxLayout()
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(320, 240)
        self.layout.addWidget(self.image_label)

        self.capture_button = QPushButton('Capture', self)
        self.capture_button.clicked.connect(self.capture_image)
        self.layout.addWidget(self.capture_button)

        self.setLayout(self.layout)
        
        self.cap = cv2.VideoCapture(0)
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

        if not os.path.exists('dataset'):
            os.makedirs('dataset')
        
        self.user_folder_name = self.get_user_folder()
        os.makedirs(self.user_folder_name)
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (320, 240))
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_img))
        
    def capture_image(self):
        if hasattr(self, 'current_frame'):
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                face = self.current_frame[y:y+h, x:x+w]
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_path = os.path.join(self.user_folder_name, f'face_{timestamp}.jpg')
                cv2.imwrite(save_path, face)
                print(f"Face saved to {save_path}")

    def get_user_folder(self):
        text, ok = QInputDialog.getText(self, 'Input Dialog', 'Enter folder name:')
        if ok and text:
            user_folder_name = os.path.join('dataset', text)
            if not os.path.exists(user_folder_name):
                return user_folder_name
            else:
                QtWidgets.QMessageBox.warning(self, "Warning", "Folder already exists. A unique name will be generated.")
                return self.get_new_user_folder()
        else:
            return self.get_new_user_folder()
        
    def get_new_user_folder(self):
        base_folder = 'dataset/New_User'
        counter = 1
        while os.path.exists(f"{base_folder}_{counter}"):
            counter += 1
        return f"{base_folder}_{counter}"
        
    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraWidget()
    window.show()
    sys.exit(app.exec_())
