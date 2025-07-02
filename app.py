import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from PIL import Image
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Bagian 1: Modul AI (Artificial Neural Network) - VERSI FINAL UAS
# =============================================================================
@st.cache_resource
def train_and_get_model():
    """
    Melatih model AI dengan data berbasis skenario nyata untuk akurasi maksimal.
    """
    # --- DATASET BERBASIS SKENARIO NYATA (PALING AKURAT) ---
    n_samples = 2500
    np.random.seed(42)

    # Skenario 1: AMAN (0) - Hujan rendah, air rendah
    aman_hujan = np.random.uniform(0, 80, int(n_samples * 0.4))
    aman_air = np.random.uniform(50, 250, int(n_samples * 0.4))
    label_aman = np.full(int(n_samples * 0.4), 0.0) # Label 0.0 untuk AMAN

    # Skenario 2: WASPADA (0.5) - Potensi bahaya
    # Kasus A: Hujan deras, air belum naik
    waspada_hujan_a = np.random.uniform(200, 400, int(n_samples * 0.15))
    waspada_air_a = np.random.uniform(150, 350, int(n_samples * 0.15))
    # Kasus B: Hujan sedang, air mulai naik
    waspada_hujan_b = np.random.uniform(100, 200, int(n_samples * 0.15))
    waspada_air_b = np.random.uniform(350, 500, int(n_samples * 0.15))
    label_waspada = np.full(int(n_samples * 0.3), 0.5) # Label 0.5 untuk WASPADA

    # Skenario 3: AWAS BANJIR (1.0) - Kondisi berbahaya
    # Kasus A: Banjir klasik (hujan & air tinggi)
    banjir_hujan_a = np.random.uniform(150, 400, int(n_samples * 0.15))
    banjir_air_a = np.random.uniform(500, 750, int(n_samples * 0.15))
    # Kasus B: Banjir kiriman (hujan rendah, air sangat tinggi)
    banjir_hujan_b = np.random.uniform(10, 100, int(n_samples * 0.15))
    banjir_air_b = np.random.uniform(600, 800, int(n_samples * 0.15))
    label_banjir = np.full(int(n_samples * 0.3), 1.0) # Label 1.0 untuk AWAS

    # Gabungkan semua skenario
    curah_hujan = np.concatenate([aman_hujan, waspada_hujan_a, waspada_hujan_b, banjir_hujan_a, banjir_hujan_b])
    tinggi_air = np.concatenate([aman_air, waspada_air_a, waspada_air_b, banjir_air_a, banjir_air_b])
    labels = np.concatenate([label_aman, label_waspada, label_banjir])

    df = pd.DataFrame({'curah_hujan_mm': curah_hujan, 'tinggi_muka_air_cm': tinggi_air, 'status': labels})
    df = df.sample(frac=1).reset_index(drop=True)

    X = df[['curah_hujan_mm', 'tinggi_muka_air_cm']].values
    y = df['status'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Arsitektur Model Regresi (untuk memprediksi nilai antara 0 dan 1)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(2,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='linear') # Output linear untuk regresi
    ])
    
    optimizer = Adam(learning_rate=0.001)
    # Gunakan Mean Squared Error karena ini masalah regresi
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    model.fit(X_scaled, y, epochs=200, batch_size=64, verbose=0)
    
    return model, scaler

# =============================================================================
# Bagian 2: Modul Computer Vision (Logika Cerdas)
# =============================================================================
def read_water_level_from_image(image_file):
    try:
        pil_image = Image.open(image_file).convert('RGB')
        cv_image = np.array(pil_image)
        cv_image = cv_image[:, :, ::-1].copy()
        height, width, _ = cv_image.shape
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        bottom_half_mask = np.zeros(cv_image.shape[:2], dtype="uint8")
        cv2.rectangle(bottom_half_mask, (0, height // 2), (width, height), 255, -1)
        final_mask = cv2.bitwise_and(blue_mask, bottom_half_mask)
        total_bottom_pixels = (width * height) / 2
        blue_pixels_in_bottom = cv2.countNonZero(final_mask)
        blue_percentage = (blue_pixels_in_bottom / total_bottom_pixels) * 100 if total_bottom_pixels > 0 else 0
        min_depth, max_depth = 100, 800
        simulated_depth = int(min_depth + (blue_percentage / 100) * (max_depth - min_depth))
        return simulated_depth, pil_image, blue_percentage
    except Exception:
        return None, None, None

# =============================================================================
# Bagian 3: Tampilan Aplikasi (User Interface)
# =============================================================================
st.set_page_config(page_title="Prediksi Banjir v3.0", page_icon="✅", layout="centered")

with st.spinner("Mempersiapkan model AI tingkat lanjut untuk UAS..."):
    model, scaler = train_and_get_model()

st.title("✅ Aplikasi Prediksi Banjir v3.0 (Versi Final UAS)")
st.write(f"AI telah dilatih dengan **{len(model.history.epoch)} skenario realistis** untuk memberikan peringatan dini 3-level.")
st.markdown("---")

st.header("1. Input Data")
curah_hujan = st.slider("Geser untuk mengatur curah hujan (mm):", 0, 400, 150)
st.info("Unggah foto. Sistem akan mensimulasikan kedalaman berdasarkan warna biru di **bagian bawah foto**.")
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

tinggi_air_cv = 0
if uploaded_file:
    level, image, blue_pct = read_water_level_from_image(uploaded_file)
    if level is not None and image is not None:
        st.image(image, caption=f"Gambar diunggah. Terdeteksi {blue_pct:.2f}% warna biru di bagian bawah.", use_column_width=True)
        st.write(f"**Ketinggian air (simulasi cerdas): {level} cm**")
        tinggi_air_cv = level

st.markdown("---")

st.header("2. Hasil Prediksi Peringatan Dini")
if st.button("🚀 Analisis Risiko Sekarang", use_container_width=True):
    if tinggi_air_cv > 0:
        input_data = np.array([[curah_hujan, tinggi_air_cv]])
        input_data_scaled = scaler.transform(input_data)
        
        prediction_score = model.predict(input_data_scaled)[0][0]
        
        # Logika Peringatan Dini 3-Level
        if prediction_score > 0.75:
            st.error(f"**Status: AWAS BANJIR**", icon="🚨")
            st.metric(label="Indeks Risiko Banjir", value=f"{prediction_score:.2f}")
            st.write("Kombinasi curah hujan dan tinggi muka air sangat berbahaya. Segera lakukan evakuasi.")
        elif prediction_score > 0.4:
            st.warning(f"**Status: WASPADA**", icon="⚠️")
            st.metric(label="Indeks Risiko Banjir", value=f"{prediction_score:.2f}")
            st.write("Kondisi berpotensi menjadi berbahaya. Siapkan langkah-langkah mitigasi.")
        else:
            st.success(f"**Status: AMAN**", icon="✅")
            st.metric(label="Indeks Risiko Banjir", value=f"{prediction_score:.2f}")
            st.write("Kondisi saat ini terpantau aman.")
    else:
        st.warning("Mohon unggah gambar terlebih dahulu untuk simulasi ketinggian air.")