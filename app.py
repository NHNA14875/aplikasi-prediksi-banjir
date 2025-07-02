import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from PIL import Image

# =============================================================================
# Bagian 1: Modul AI (Artificial Neural Network)
# =============================================================================
@st.cache_resource
def train_and_get_model():
    """
    Melatih dan menyimpan model AI sederhana menggunakan cache Streamlit.
    """
    # Data historis simulasi
    data = {
        'curah_hujan_mm': [50, 80, 120, 150, 180, 200, 220, 250, 270, 300, 320],
        'tinggi_muka_air_cm': [150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650],
        'terjadi_banjir': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1] # 0=Aman, 1=Banjir
    }
    df = pd.DataFrame(data)
    
    X = df[['curah_hujan_mm', 'tinggi_muka_air_cm']].values
    y = df['terjadi_banjir'].values
    
    # Membuat model ANN
    model = Sequential([
        Dense(16, activation='relu', input_shape=(2,)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=100, verbose=0)
    
    return model

# =============================================================================
# Bagian 2: Modul Computer Vision (Simulasi)
# =============================================================================
def read_water_level_from_image(image_file):
    """
    Mensimulasikan pembacaan tinggi muka air dari gambar.
    """
    try:
        img = Image.open(image_file)
        # Simulasi: menghasilkan angka acak untuk tinggi air
        simulated_level = np.random.randint(100, 700)
        return simulated_level, img
    except Exception as e:
        st.error(f"Gagal memproses gambar: {e}")
        return None, None

# =============================================================================
# Bagian 3: Tampilan Aplikasi (User Interface)
# =============================================================================

st.set_page_config(page_title="Prediksi Banjir", page_icon="💧", layout="centered")

# Memuat model AI
with st.spinner("Mempersiapkan model AI..."):
    model = train_and_get_model()

st.title("💧 Aplikasi Prototipe Prediksi Banjir")
st.write(f"Aplikasi ini dibuat berdasarkan proposal **'Prediksi Banjir Berbasis Machine Learning'** oleh **Yobby Azriel Iqdhi Vianta (NIM: A11.2023.14890)**.")
st.markdown("---")

# Kolom Input
st.header("1. Input Data")

# Input Curah Hujan
curah_hujan = st.slider("Geser untuk mengatur curah hujan (mm):", 0, 400, 150)

# Input Tinggi Muka Air via Computer Vision
st.info("Unggah foto papan duga air untuk simulasi pembacaan otomatis.")
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

tinggi_air_cv = 0
if uploaded_file:
    level, image = read_water_level_from_image(uploaded_file)
    if level and image:
        st.image(image, caption=f"Gambar diunggah. Ketinggian air (simulasi): {level} cm", use_column_width=True)
        tinggi_air_cv = level

st.markdown("---")

# Tombol dan Hasil Prediksi
st.header("2. Hasil Prediksi")
if st.button("🚀 Lakukan Prediksi Sekarang", use_container_width=True):
    if tinggi_air_cv > 0:
        input_data = np.array([[curah_hujan, tinggi_air_cv]])
        prediction_proba = model.predict(input_data)[0][0]
        
        if prediction_proba > 0.5:
            st.error(f"**Status: AWAS BANJIR**")
            st.metric(label="Tingkat Keyakinan Model", value=f"{prediction_proba:.0%}")
        else:
            st.success(f"**Status: AMAN**")
            st.metric(label="Tingkat Keyakinan Model", value=f"{1-prediction_proba:.0%}")
    else:
        st.warning("Mohon unggah gambar papan duga air terlebih dahulu.")
