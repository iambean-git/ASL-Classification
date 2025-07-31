import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# 모델 블록 정의
class MyConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.model(x)

# ASL 모델 아키텍처
IMG_CHS = 1
IMG_WIDTH, IMG_HEIGHT = 28, 28
N_CLASSES = 24  

class ASLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            MyConvBlock(IMG_CHS, 25, 0.0),
            MyConvBlock(25, 50, 0.2),
            MyConvBlock(50, 75, 0.0),
            nn.Flatten(),
            nn.Linear(75 * 3 * 3, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512, N_CLASSES)
        )

    def forward(self, x):
        return self.net(x)

# 레이블 매핑 
label_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
    10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P',
    15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U',
    20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

# 장치 설정 및 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ASLModel().to(device)
model_path = './model.pth'
model = torch.load(model_path, map_location=device , weights_only=False).to(device)
model.eval()

# 전처리 정의
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Streamlit UI
st.title("ASL Alphabet Prediction")
st.write("이미지를 업로드하거나 카메라로 촬영하여 ASL 알파벳을 예측합니다.")

uploaded_file = st.file_uploader("Upload an ASL image", type=['png', 'jpg', 'jpeg'])
captured = st.camera_input("Or capture with your camera")

input_image = None
if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
elif captured is not None:
    # 카메라 입력은 좌우가 뒤집혀 오므로 모델 학습 시 방향과 맞추기 위해 반전
    input_image = Image.open(captured).transpose(Image.FLIP_LEFT_RIGHT)
    
    raw = Image.open(captured).transpose(Image.FLIP_LEFT_RIGHT)
    # 중앙 사각형 영역으로 크롭하여 손 부분을 강조
    w, h = raw.size
    size = min(w, h)
    left = (w - size) // 2
    top = (h - size) // 2
    cropped = raw.crop((left, top, left + size, top + size))
    input_image = cropped

if input_image is not None:
    st.image(input_image, caption="Input Image", use_container_width=True)
    if st.button("Predict"):
        img_tensor = preprocess(input_image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            pred_label = label_map[int(pred_idx)]
            st.markdown(
                f"<h2>Predicted: <strong>{pred_label}</strong> ({float(confidence)*100:.2f}%)</h2>",
                unsafe_allow_html=True
            )