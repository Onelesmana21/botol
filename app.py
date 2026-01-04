import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import av
import cv2
from PIL import Image

# Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Minuman Real-time", layout="wide")

st.title("🥤 Aplikasi Deteksi Minuman Kemasan")
st.write("Mendeteksi Botol, Kaleng, dan Kotak secara Real-time.")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Pastikan file best.pt sudah diupload ke GitHub
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# --- LOGIKA VIDEO PROCESSING (REAL-TIME) ---
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Proses deteksi frame demi frame
        # conf=0.4 (hanya tampilkan jika tingkat keyakinan di atas 40%)
        results = model.predict(img, conf=0.4, verbose=False)
        
        # Gambar bounding box hasil prediksi ke frame
        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- MENU UTAMA ---
tabs = st.tabs(["🎥 Live Kamera", "📂 Upload Gambar"])

with tabs[0]:
    st.subheader("Live Streaming Detection")
    st.info("Klik 'Start' di bawah untuk menyalakan kamera.")
    
    # Menjalankan WebRTC Streamer
    webrtc_streamer(
        key="yolo-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with tabs[1]:
    st.subheader("Deteksi via Upload")
    uploaded_file = st.file_uploader("Upload foto minuman...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar Asli', width=400)
        
        # Jalankan Prediksi
        results = model.predict(image)
        res_plotted = results[0].plot()
        
        st.image(res_plotted, caption='Hasil Deteksi', use_container_width=True)
        
        # Tampilkan list objek yang ditemukan
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            st.write(f"- **{model.names[cls_id]}** (Confidence: {conf:.2f})")
