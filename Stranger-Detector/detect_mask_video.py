import cv2
import numpy as np
import pickle
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import time

# LabelEncoder 객체 로드
with open('label_encoder.pkl', 'rb') as f:
    lb = pickle.load(f)

def detect_and_predict_celebrity(frame, faceNet, celebrityNet, lb):
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

            # 검출된 얼굴 영역이 유효한지 확인
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

    return (locs, preds)

def generate_frames():
    # 얼굴 탐지 모델 로드
    prototxtPath = "face_detector/deploy.prototxt"
    weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # 셀러브리티 인식 모델 로드 (H5 형식)
    celebrityNet = load_model("celebrity_detector.h5")

    # 웹캠 비디오 스트림 시작
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        (locs, preds) = detect_and_predict_celebrity(frame, faceNet, celebrityNet, lb)
        
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)  # 검출된 얼굴 영역 표시
            max_index = np.argmax(pred)
            label = lb.classes_[max_index]
            confidence = pred[max_index]

            # 예측 확률이 0.4 이상일 경우에만 레이블 표시
            if confidence > 0.4:
                color = (0, 255, 0)
                label = "{}: {:.2f}%".format(label, confidence * 100)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # 프레임을 JPEG로 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # 프레임을 multipart 형식으로 전송
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
