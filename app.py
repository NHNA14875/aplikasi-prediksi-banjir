import streamlit as st
import pandas as pd
import numpy as np
import cv2 # OpenCV untuk Computer Vision
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from PIL import Image
import os

# =============================================================================
# Bagian 1: Modul AI (Artificial Neural Network) - Sesuai Paper
# =============================================================================

# Fungsi ini akan di-cache oleh Streamlit, artinya model hanya akan dilatih sekali.
@st.cache_resource
def train_and_get_model():
    """
    Fungsi ini melatih model AI sederhana berdasarkan data CSV.
    Model ini adalah ANN seperti yang diusulkan paper.
    """
    st.info("Mempersiapkan model AI untuk pertama kali. Mohon tunggu...")
    
    # Data historis (meniru data dari BMKG & BPBD)
    data = {
        'curah_hujan_mm': [50, 80, 120, 150, 180, 200, 220, 250, 270, 300],
        'tinggi_muka_air_cm': [150, 200, 250, 300, 350, 400, 450, 500, 550, 600],
        'terjadi_banjir': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1] # 0 = Aman, 1 = Banjir
    }
    df = pd.DataFrame(data)
    
    # Memisahkan fitur (input) dan label (output)
    X = df[['curah_hujan_mm', 'tinggi_muka_air_cm']].values
    y = df['terjadi_banjir'].values
    
    # Membuat model ANN sederhana sesuai usulan paper
    model = Sequential([
        Dense(10, activation='relu', input_shape=(2,)),
        Dense(5, activation='relu'),
        Dense(1, activation='sigmoid') # Sigmoid untuk output biner (0 atau 1)
    ])
    
    # Meng-compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Melatih model
    model.fit(X, y, epochs=100, verbose=0) # verbose=0 agar tidak menampilkan proses training di terminal
    
    st.success("Model AI siap digunakan!")
    return model

# =============================================================================
# Bagian 2: Modul Computer Vision
# =============================================================================
def read_water_level_from_image(image_file):
    """
    Fungsi ini mensimulasikan pembacaan tinggi muka air dari gambar papan duga.
    Ini adalah implementasi visi komputer untuk otomatisasi input data.
    """
    try:
        # Mengubah file gambar menjadi format yang bisa dibaca OpenCV
        img = Image.open(image_file)
        img_cv = np.array(img)
        
        # --- LOGIKA SIMULASI SEDERHANA ---
        # Dalam aplikasi nyata, di sini akan ada kode deteksi warna biru (air)
        # dan pembacaan angka pada papan duga.
        # Untuk prototipe ini, kita gunakan nilai acak sebagai simulasi.
        simulated_level = np.random.randint(100, 550)
        
        return simulated_level, img
    except Exception as e:
        st.error(f"Error saat memproses gambar: {e}")
        return None, None


# =============================================================================
# Bagian 3: Tampilan Aplikasi (User Interface) dengan Streamlit
# =============================================================================

st.set_page_config(page_title="Prediksi Banjir", page_icon="💧", layout="wide")

# Memanggil fungsi untuk melatih dan mendapatkan model
model = train_and_get_model()

st.title("💧 Aplikasi Prototipe Prediksi Banjir")
st.write(f"Aplikasi ini dibuat berdasarkan proposal **'Prediksi Banjir Berbasis Machine Learning'** oleh **Yobby Azriel Iqdhi Vianta (NIM: A11.2023.14890)**.")
st.markdown("---")

# Membuat dua kolom untuk tata letak yang lebih baik
col1, col2 = st.columns(2)

with col1:
    st.header("1. Input Data")
    st.write("Masukkan data cuaca dan kondisi air saat ini.")
    
    # --- Input dari Computer Vision ---
    st.subheader("A. Input Tinggi Muka Air (via Computer Vision)")
    st.info("Unggah foto papan duga air untuk mendapatkan data ketinggian secara otomatis.")
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    
    tinggi_air_cv = 0
    if uploaded_file is not None:
        level, image = read_water_level_from_image(uploaded_file)
        if level and image:
            st.image(image, caption=f"Gambar yang diunggah. Ketinggian air (simulasi): {level} cm", use_column_width=True)
            tinggi_air_cv = level

    # --- Input Manual ---
    st.subheader("B. Input Curah Hujan (Data dari BMKG)")
    curah_hujan = st.slider("Geser untuk mengatur curah hujan (mm):", min_value=0, max_value=400, value=150)

with col2:
    st.header("2. Hasil Prediksi")
    st.write("Klik tombol di bawah untuk melihat hasil prediksi dari model AI.")
    
    # Tombol untuk melakukan prediksi
    if st.button("🚀 Lakukan Prediksi Sekarang", use_container_width=True):
        if tinggi_air_cv > 0:
            # Menyiapkan data untuk prediksi
            input_data = np.array([[curah_hujan, tinggi_air_cv]])
            
            # Melakukan prediksi dengan model AI
            prediction_proba = model.predict(input_data)[0][0]
            
            st.subheader("Hasil Analisis Model AI:")
            
            if prediction_proba > 0.6: # Batas probabilitas (threshold) bisa diubah
                st.error(f"**Status: AWAS BANJIR**")
                st.metric(label="Tingkat Keyakinan Model", value=f"{prediction_proba:.0%}")
                st.write("Sistem memprediksi potensi banjir **TINGGI** berdasarkan data yang dimasukkan. Segera lakukan langkah mitigasi sesuai prosedur.")
            else:
                st.success(f"**Status: AMAN**")
                st.metric(label="Tingkat Keyakinan Model", value=f"{1-prediction_proba:.0%}")
                st.write("Berdasarkan data saat ini, potensi banjir **RENDAH**.")
        else:
            st.warning("Mohon unggah gambar papan duga air terlebih dahulu di kolom sebelah kiri.")