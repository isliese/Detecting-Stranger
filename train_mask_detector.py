# 필요한 패키지 임포트
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# 초기 학습률, 에포크 수, 배치 크기 설정
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# 데이터셋 디렉토리와 카테고리 설정
DIRECTORY = "./dataset"
CATEGORIES = ["ChaEunwoo", "JangWonyoung", "Stranger"]  # 'others'에서 'Stranger'로 변경

# 데이터와 레이블을 초기화하고 이미지 로드
print("[INFO] 이미지 로딩 중...")

data = []
labels = []

# 이미지 경로를 순회하며 이미지 로드 및 전처리
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        # .DS_Store 파일 무시
        if img == ".DS_Store":
            continue
        try:
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)

            data.append(image)
            labels.append(category)
        except Exception as e:
            print(f"[WARN] 이미지 로드 실패: {img_path}, 오류: {e}")

# 레이블 원-핫 인코딩
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = np.array(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

# 데이터 차원과 레이블 확인
print(f"데이터 차원: {data.shape}")
print(f"레이블 차원: {labels.shape}")

# 데이터를 학습 및 테스트 세트로 분할
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# 데이터 증강 설정
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# MobileNetV2 모델 로드
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# 최상단 모델 구성
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(CATEGORIES), activation="softmax")(headModel)  # 카테고리 수에 맞게 변경

# 최상단 모델을 기본 모델 위에 쌓기
model = Model(inputs=baseModel.input, outputs=headModel)

# 기본 모델의 모든 레이어를 고정하여 학습되지 않도록 설정
for layer in baseModel.layers:
    layer.trainable = False

# 모델 컴파일
print("[INFO] 모델 컴파일 중...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# 모델 학습
print("[INFO] 모델 학습 중...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# 모델 평가
print("[INFO] 모델 평가 중...")
predIdxs = model.predict(testX, batch_size=BS)

# 각 테스트 이미지에 대해 가장 높은 예측 확률을 갖는 레이블의 인덱스 찾기
predIdxs = np.argmax(predIdxs, axis=1)

# 분류 보고서 출력
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# 모델 저장
print("[INFO] 모델 저장 중...")
model.save("celebrity_detector.model", save_format="h5")

# 학습 손실 및 정확도 시각화
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
