import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Minuman", page_icon="🥤")

# Judul Utama
st.title("AI Deteksi Kemasan Minuman 🥤")
st.write("Aplikasi ini mendeteksi Botol, Kaleng, dan Kotak menggunakan YOLOv8.")

# 1. Load Model (Gunakan Cache agar tidak lemot)
@st.cache_resource
def load_model():
    # Pastikan file 'best.pt' ada di folder yang sama di GitHub
    model = YOLO('best.pt') 
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model 'best.pt'. Pastikan file sudah di-upload ke GitHub. Error: {e}")
    st.stop()

# 2. Sidebar Navigasi
st.sidebar.title("Menu Utama")
mode = st.sidebar.selectbox("Pilih Metode Deteksi:", ["Upload Gambar", "Gunakan Kamera"])

# 3. Logika Deteksi
if mode == "Upload Gambar":
    img_file = st.file_uploader("Pilih file gambar (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
    if img_file is not None:
        image = Image.open(img_file)
        
        # Jalankan Prediksi
        with st.spinner('Menganalisis...'):
            results = model.predict(image, conf=0.25)
            res_plotted = results[0].plot() # Hasil dengan bounding box
            
        st.image(res_plotted, caption="Hasil Deteksi", use_container_width=True)
        
        # Tampilkan Detail Objek
        for box in results[0].boxes:
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]
            st.write(f"✅ Terdeteksi: **{label}** (Tingkat Keyakinan: {conf:.2f})")

elif mode == "Gunakan Kamera":
    img_cam = st.camera_input("Ambil foto objek minuman")
    if img_cam is not None:
        image = Image.open(img_cam)
        
        results = model.predict(image, conf=0.25)
        res_plotted = results[0].plot()
        
        st.image(res_plotted, caption="Hasil Deteksi Kamera", use_container_width=True)
