import streamlit as st
from ultralytics import YOLO
import PIL.Image
import numpy as np
import cv2

st.title("Deteksi Minuman Kemasan 🥤")
st.sidebar.title("Pengaturan")

# Load Model (Gunakan hasil export tadi)
model_path = 'best.pt' # Ganti dengan model hasil trainingmu
try:
    model = YOLO(model_path)
except:
    st.error("Model tidak ditemukan. Silakan training dulu di Notebook.")

mode = st.sidebar.selectbox("Pilih Mode", ["Upload File", "Real-time Kamera"])

if mode == "Upload File":
    img_file = st.file_uploader("Upload Gambar Minuman (Botol/Kaleng/Kotak)", type=['jpg', 'png', 'jpeg'])
    if img_file:
        img = PIL.Image.open(img_file)
        results = model.predict(img)
        
        # Gambar bounding box
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Hasil Deteksi", use_column_width=True)

elif mode == "Real-time Kamera":
    img_file = st.camera_input("Ambil Foto untuk Deteksi")
    if img_file:
        img = PIL.Image.open(img_file)
        results = model.predict(img)
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Hasil Deteksi Kamera")
