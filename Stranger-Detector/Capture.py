import sys
import cv2
import os
import datetime
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton, QInputDialog, QMessageBox
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

class CameraWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        # 윈도우 제목과 크기 설정
        self.setWindowTitle('Camera')
        self.setGeometry(50, 50, 320, 240)

        # 레이아웃과 라벨 설정
        self.layout = QVBoxLayout()
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(320, 240)
        self.layout.addWidget(self.image_label)

        # 캡처 버튼 설정
        self.capture_button = QPushButton('Start Capture', self)
        self.capture_button.clicked.connect(self.start_capture)
        self.layout.addWidget(self.capture_button)

        self.setLayout(self.layout)
        
        # 카메라 초기화
        self.cap = cv2.VideoCapture(0)
        
        # 얼굴 인식용 Haar Cascade 로드
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 타이머 설정 (20ms마다 프레임 업데이트)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

        # 데이터셋 폴더가 없으면 생성
        if not os.path.exists('dataset'):
            os.makedirs('dataset')
        
        # 사용자 폴더 이름 설정 및 생성
        self.user_folder_name = self.get_user_folder()
        os.makedirs(self.user_folder_name)
        
        # 캡처 카운터 초기화
        self.capture_count = 0
        
    def update_frame(self):
        # 카메라 프레임 읽기
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                # 얼굴에 사각형 그리기
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # 프레임을 RGB로 변환하고 크기 조정
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (320, 240))
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_img))
        
    def start_capture(self):
        # 캡처 버튼 비활성화 및 캡처 타이머 시작
        self.capture_button.setEnabled(False)
        self.capture_count = 0
        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self.capture_image)
        self.capture_timer.start(250)  # 30초 동안, 1초에 4장 캡처, 총 120장
        
    def capture_image(self):
        if self.capture_count < 120:
            if hasattr(self, 'current_frame'):
                gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    # 얼굴 영역을 잘라서 저장
                    face = self.current_frame[y:y+h, x:x+w]
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    save_path = os.path.join(self.user_folder_name, f'face_{timestamp}.jpg')
                    cv2.imwrite(save_path, face)
                    print(f"Face saved to {save_path}")
                    
            self.capture_count += 1
        else:
            # 30장 캡처 완료 후 타이머 중지 및 버튼 활성화
            self.capture_timer.stop()
            self.capture_button.setEnabled(True)
            QMessageBox.information(self, "Information", "120 images captured.")

    def get_user_folder(self):
        # 사용자에게 폴더 이름 입력받기
        text, ok = QInputDialog.getText(self, 'Input Dialog', '폴더 이름을 입력해주세요:')
        if ok and text:
            user_folder_name = os.path.join('dataset', text)
            if not os.path.exists(user_folder_name):
                return user_folder_name
            else:
                QMessageBox.warning(self, "Warning", "Folder already exists. A unique name will be generated.")
                return self.get_new_user_folder()
        else:
            return self.get_new_user_folder()
        
    def get_new_user_folder(self):
        # 새로운 사용자 폴더 이름 생성
        base_folder = 'dataset/New_User'
        counter = 1
        while os.path.exists(f"{base_folder}_{counter}"):
            counter += 1
        return f"{base_folder}_{counter}"
        
    def closeEvent(self, event):
        # 윈도우 닫힐 때 카메라와 모든 창 닫기
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraWidget()
    window.show()
    sys.exit(app.exec_())
