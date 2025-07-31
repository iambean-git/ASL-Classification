# ✌ CNN 기반의 미국 수화 (American Sign Language) 분류 모델 개발 및 성능 평가

## 💻 개발환경 및 실행
- requirements.txt 참고
- 실행 방법
  ``` bash
  streamlit run asl_app.py
  ```
  
<br>

## 🌀 문제정의

- American Sign Language (ASL) 알파벳 분류를 위한 이미지 데이터셋 활용
    - 손으로 표현한 알파벳 이미지들을 학습하여 어떤 문자인지를 예측하는 모델 개발
    - Kaggle의 Sign Language MNIST 데이터 셋을 이용
- CNN 기반 모델 설계 및 학습
    - 합성곱 신경망(Convolutional Neural Network)을 사용하여 이미지 특징을 추출하고 분류
- 데이터 증강(Augmentation) 기법 적용
    - 회전, 이동, 크기 조절 등다양한 변형을 통해 데이터 다양성 확보
    - 일반화 성능 향상 및 과적합 방지
- 정확도 및 Confusion Matrix 기반 성능 평가
- Streamlit을 활용한 AI 앱 개발

<br>

## 💾 데이터셋
- Kaggle의 Sign Language MNIST 데이터셋 활용 ([출처](https://www.kaggle.com/datasets/datamunge/sign-language-mnist))
- 24개의 클래스
    - J 와 Z를 제외한 A– Z 까지의 알파벳
    - J, Z 두 글자는 움직임이 필요해 훈련 데이터셋에 포함되지 않는다
  <img width="600" alt="image" src="https://github.com/user-attachments/assets/9be8c657-2dd3-4d09-ae29-eba62ea8ca64" />

<br><br>

## 🔧 모델 아키텍처
<img width="200" alt="image" src="https://github.com/user-attachments/assets/4fee558d-0cbc-4c36-97e3-b050a9d60190" />

- **CNN 기반 커스텀 모델**
    - 합성곱 신경망 (CNN) 아키텍처 구현
    - 입력 이미지 : (1, 28, 28) (Grayscale, 크기 통일)
    - Conv2D → ReLU → MaxPool → Dropout → Linear 구성
    - 적절한 Regularization을 통한 과적합 방지
- **데이터 증강 (Data Augmentation)**
``` python
random_transforms = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), scale=(.9, 1), ratio=(1, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=.2, contrast=.5)
])
```

<br>

## 📊 성능 평가
### 1) train, validation 데이터에 대한 loss 및 accuracy 값을 통한 과적합 확인
<img width="80%" height="659" alt="image" src="https://github.com/user-attachments/assets/9e7d90c4-54f5-4781-a9d6-801a704a36cd" />

### 2) Confusion Matrix
<img width="50%" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/7526c098-533f-4096-9ade-e34e74e69d77" />

### 3) Precision, Recall, F1-Score
<img width="50%" alt="image" src="https://github.com/user-attachments/assets/78d88ef9-8d9c-48a8-88d7-00adb4b5e3d4" />


<br><br>


## ✨ Streamlit을 사용한 웹앱 구현

<img width="31%" alt="test_X" src="https://github.com/user-attachments/assets/5c773292-3806-42d5-bb66-1d6b2df70e50" />
<img width="31%" alt="test_U" src="https://github.com/user-attachments/assets/58d02e11-c530-4138-86f3-20fe1701b3a2" />
<img width="31%" alt="test_W" src="https://github.com/user-attachments/assets/0e585de3-1785-42dd-991d-6548583b76d1" />


