import cv2
import numpy as np
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import time
from PIL import ImageFont, ImageDraw, Image
import custom_layers

# 시스템 폰트 경로
system_font_path = "/Users/yeonjin/Downloads/Nanum_Gothic_Coding/NanumGothicCoding-Regular.ttf"

# 데이터셋 폴더 기반으로 클래스 목록 설정
dataset_dir = "dataset"
labels = os.listdir(dataset_dir)
folder_names = {i: label for i, label in enumerate(labels)}

# 얼굴 탐지 모델 및 셀러브리티 인식 모델 로드
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
celebrityNet = load_model("celebrity_detector.h5", custom_objects={'DepthwiseConv2D': custom_layers.CustomDepthwiseConv2D})

def detect_and_predict_celebrity(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            if endX - startX > 0 and endY - startY > 0:
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = celebrityNet.predict(faces, batch_size=32)
        print(f"Predictions: {preds}")  # 디버깅: 예측 결과 출력
    else:
        print("No faces detected.")

    return (locs, preds)

def put_korean_text(image, text, position, font_path=system_font_path, font_size=20, color=(255, 255, 255)):
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        print(f"Warning: '{font_path}' 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        font = ImageFont.load_default()

    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    draw.text(position, text, font=font, fill=color)
    image = np.array(image_pil)
    return image

def generate_frames():
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        (locs, preds) = detect_and_predict_celebrity(frame)
        
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)  # 검출된 얼굴 영역 표시
            max_index = np.argmax(pred)
            confidence = pred[max_index]

            label = folder_names.get(max_index, "미등록자")
            
            if label == "미등록자":
                color = (0, 0, 255)  # 빨간색
            else:
                color = (0, 255, 0)  # 녹색

            label_text = "{}: {:.2f}%".format(label, confidence * 100)
            print(f"Label: {label_text}")  # 디버깅: 레이블 텍스트 출력
            frame = put_korean_text(frame, label_text, (startX, startY - 30), font_size=20, color=color)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # Frame encoding
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    vs.stop()
