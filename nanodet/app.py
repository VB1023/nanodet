import streamlit as st
import torch
from PIL import Image
from nanodet.model import Nanodet  # Adjust this based on the actual structure
from nanodet.util import load_model  # Adjust this based on the actual structure

# Load the pre-trained model
model = Nanodet()  # Adjust parameters as necessary
model.load_state_dict(torch.load('workspace/nanodet-plus-m_416/model_best/nanodet_model_best.pth'))
model.eval()

st.title("NanoDet Object Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image and make predictions
    results = model.predict(image)  # Adjust this based on the actual prediction method
    st.write("Predictions:", results)
    
    
    
