import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import torch
from model import FaceRecognitionModel
from pytorch_metric_learning.losses import ArcFaceLoss
import albumentations as A

transform_inf = A.Compose([
    A.Resize(128, 128),
    A.pytorch.ToTensorV2()
])

# Принудительно устанавливаем тёмную тему
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def load_model(path = "model/model.pth"):
    model = FaceRecognitionModel()
    model.init(ArcFaceLoss(500, 512), 512)
    model.load_state_dict(torch.load(path)['model'])
    model.eval()
    return model

st.title("Классификация лиц на основе facenet")
st.write("Загрузите изображение, и модель определит класс")
uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_container_width = True)

    if st.button("Классифицировать"):
        model = load_model()
        image = transform_inf(image=np.array(image))['image'][np.newaxis, :, :, :] / 256
        predictions = model.predict_proba(torch.Tensor(image)).detach().cpu()[0]
        predictions = torch.round(predictions, decimals=4) * 100
        confidence = predictions.max()
        st.success(f"Результат: Class {predictions.argmax():3d} (Уверенность: {confidence:.2f})")
        
        args = np.argsort(predictions)[-6:-1]
        args = torch.flip(args, dims=(-1,))
        data = pd.DataFrame({'Class': args, 'Probability': predictions[args]})
        st.table(data)