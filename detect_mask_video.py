# 필요한 패키지 임포트
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_celebrity(frame, faceNet, celebrityNet):
    # 프레임의 크기 가져오기
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    # 네트워크를 통해 얼굴 검출
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # 얼굴, 위치 및 예측 리스트 초기화
    faces = []
    locs = []
    preds = []

    # 검출된 얼굴들에 대한 반복문
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

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

prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

celebrityNet = load_model("celebrity_detector.model")

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (locs, preds) = detect_and_predict_celebrity(frame, faceNet, celebrityNet)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (chaEunwoo, jangWonyoung, stranger) = pred

        if chaEunwoo > jangWonyoung and chaEunwoo > stranger and chaEunwoo > 0.5:
            label = "ChaEunwoo"
            color = (0, 255, 0)
        elif jangWonyoung > chaEunwoo and jangWonyoung > stranger and jangWonyoung > 0.5:
            label = "JangWonyoung"
            color = (0, 255, 0)
        else:
            label = "Stranger"
            color = (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(chaEunwoo, jangWonyoung, stranger) * 100)
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
