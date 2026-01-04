import streamlit as st
import os

# Pengaturan Lingkungan (harus di paling atas sebelum import cv2/ultralytics)
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

try:
    import cv2
    import av
    import numpy as np
    from PIL import Image
    from ultralytics import YOLO
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
except ImportError as e:
    st.error(f"Gagal memuat library: {e}")
    st.info("Pastikan requirements.txt sudah sesuai dan tunggu proses instalasi selesai.")
    st.stop()

# Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Minuman AI", layout="centered")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Pastikan file best.pt ada di root repository GitHub Anda
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"File model 'best.pt' tidak ditemukan! Pastikan sudah diunggah ke GitHub. Error: {e}")
    st.stop()

# --- HEADER ---
st.title("🥤 Deteksi Kemasan Minuman")
st.write("Aplikasi deteksi otomatis untuk Botol, Kaleng, dan Kotak.")

# --- LOGIKA VIDEO PROCESSING (WEB RTC) ---
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Inisialisasi deteksi YOLO
        # imgsz=320 untuk mempercepat proses di CPU Cloud
        results = self.model.predict(img, conf=0.4, imgsz=320, verbose=False)
        
        # Gambar hasil deteksi ke frame
        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- NAVIGASI TAB ---
tab1, tab2 = st.tabs(["🎥 Kamera Real-time", "📂 Upload Gambar"])

with tab1:
    st.subheader("Live Detection")
    st.info("Klik 'Start' untuk mengaktifkan kamera.")
    
    # Konfigurasi Server STUN (Penting agar kamera jalan di Cloud)
    RTC_CONFIG = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="yolo-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with tab2:
    st.subheader("Manual Upload")
    uploaded_file = st.file_uploader("Pilih gambar...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Jalankan Prediksi
        with st.spinner('Menganalisis gambar...'):
            results = model.predict(image, conf=0.4)
            res_plotted = results[0].plot()
            
            # Tampilkan Gambar
            st.image(res_plotted, caption='Hasil Deteksi', use_column_width=True)
            
            # Tampilkan Hasil Detail
            boxes = results[0].boxes
            if len(boxes) > 0:
                st.write(f"Ditemukan {len(boxes)} objek:")
                for box in boxes:
                    label = model.names[int(box.cls[0])]
                    prob = float(box.conf[0])
                    st.write(f"- **{label}** (Keyakinan: {prob:.2f})")
            else:
                st.warning("Tidak ada objek yang terdeteksi.")

# --- FOOTER ---
st.markdown("---")
st.caption("Aplikasi berbasis YOLOv8 & Streamlit")
