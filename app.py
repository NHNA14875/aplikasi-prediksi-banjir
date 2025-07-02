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
# Bagian 1: Modul AI (Artificial Neural Network) - VERSI SUPER CERDAS
# =============================================================================
@st.cache_resource
def train_and_get_model():
    """
    Melatih dan menyimpan model AI yang jauh lebih pintar dengan data yang masif dan kompleks.
    """
    st.info("Mempersiapkan model AI tingkat lanjut. Proses ini mungkin memakan waktu lebih lama saat pertama kali dijalankan...")

    # --- PENCIPTAAN DATASET RAKSASA (PENGGANTI DATA SEDERHANA) ---
    # Kita akan membuat 2000 skenario data yang realistis.
    n_samples = 2000
    np.random.seed(42) # Agar hasil data selalu sama

    # Skenario 1: Kondisi Aman (Curah hujan & tinggi air rendah)
    aman_hujan = np.random.uniform(0, 150, int(n_samples * 0.6))
    aman_air = np.random.uniform(50, 350, int(n_samples * 0.6))
    aman_label = np.zeros(int(n_samples * 0.6))

    # Skenario 2: Potensi Banjir (Kombinasi tinggi)
    banjir_hujan = np.random.uniform(150, 400, int(n_samples * 0.3))
    banjir_air = np.random.uniform(400, 750, int(n_samples * 0.3))
    banjir_label = np.ones(int(n_samples * 0.3))

    # Skenario 3: Anomali (Hujan rendah, air tinggi -> Banjir kiriman)
    anomali_hujan_1 = np.random.uniform(10, 100, int(n_samples * 0.05))
    anomali_air_1 = np.random.uniform(500, 750, int(n_samples * 0.05))
    anomali_label_1 = np.ones(int(n_samples * 0.05))

    # Skenario 4: Anomali (Hujan tinggi, air rendah -> Tidak banjir)
    anomali_hujan_2 = np.random.uniform(250, 400, int(n_samples * 0.05))
    anomali_air_2 = np.random.uniform(100, 300, int(n_samples * 0.05))
    anomali_label_2 = np.zeros(int(n_samples * 0.05))

    # Gabungkan semua skenario menjadi satu dataset besar
    curah_hujan = np.concatenate([aman_hujan, banjir_hujan, anomali_hujan_1, anomali_hujan_2])
    tinggi_air = np.concatenate([aman_air, banjir_air, anomali_air_1, anomali_air_2])
    label_banjir = np.concatenate([aman_label, banjir_label, anomali_label_1, anomali_label_2])

    df = pd.DataFrame({
        'curah_hujan_mm': curah_hujan,
        'tinggi_muka_air_cm': tinggi_air,
        'terjadi_banjir': label_banjir
    })

    # Acak data agar urutannya tidak tertebak
    df = df.sample(frac=1).reset_index(drop=True)
    
    X = df[['curah_hujan_mm', 'tinggi_muka_air_cm']].values
    y = df['terjadi_banjir'].values

    # Normalisasi data sangat penting untuk model yang lebih dalam
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- ARSITEKTUR OTAK AI YANG LEBIH DALAM ---
    model = Sequential([
        Dense(64, activation='relu', input_shape=(2,)),
        Dropout(0.3), # Mencegah AI "terlalu hafal"
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Latih model dengan data yang jauh lebih banyak
    model.fit(X_scaled, y, epochs=200, batch_size=64, verbose=0)
    
    # Simpan juga scaler yang sudah dilatih, ini sangat penting
    return model, scaler

# =============================================================================
# Bagian 2: Modul Computer Vision (Logika Cerdas)
# =============================================================================
def read_water_level_from_image(image_file):
    """
    Mensimulasikan pembacaan tinggi muka air dengan menganalisis
    persentase warna biru HANYA DI BAGIAN BAWAH GAMBAR.
    """
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
        
        if total_bottom_pixels == 0:
            blue_percentage = 0
        else:
            blue_percentage = (blue_pixels_in_bottom / total_bottom_pixels) * 100

        min_depth = 100
        max_depth = 700
        simulated_depth = int(min_depth + (blue_percentage / 100) * (max_depth - min_depth))
        
        return simulated_depth, pil_image, blue_percentage

    except Exception:
        return None, None, None

# =============================================================================
# Bagian 3: Tampilan Aplikasi (User Interface)
# =============================================================================

st.set_page_config(page_title="Prediksi Banjir v2.0", page_icon="🧠", layout="centered")

# Memuat model AI dan scaler
model, scaler = train_and_get_model()

st.title("🧠 Aplikasi Prediksi Banjir v2.0 (AI Super Cerdas)")
st.write(f"Versi ini menggunakan AI yang dilatih pada **2000 skenario data kompleks**.")
st.markdown("---")

st.header("1. Input Data")

curah_hujan = st.slider("Geser untuk mengatur curah hujan (mm):", 0, 400, 150)

st.info("Unggah foto. Sistem akan mensimulasikan kedalaman berdasarkan jumlah warna biru di **bagian bawah foto**.")
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

tinggi_air_cv = 0
if uploaded_file:
    level, image, blue_pct = read_water_level_from_image(uploaded_file)
    if level is not None and image is not None:
        st.image(image, caption=f"Gambar diunggah. Terdeteksi {blue_pct:.2f}% warna biru di bagian bawah.", use_column_width=True)
        st.write(f"**Ketinggian air (simulasi cerdas): {level} cm**")
        tinggi_air_cv = level

st.markdown("---")

st.header("2. Hasil Prediksi")
if st.button("🚀 Lakukan Prediksi Sekarang", use_container_width=True):
    if tinggi_air_cv > 0:
        # PENTING: Gunakan scaler yang sama untuk mengubah data input baru
        input_data = np.array([[curah_hujan, tinggi_air_cv]])
        input_data_scaled = scaler.transform(input_data)
        
        prediction_proba = model.predict(input_data_scaled)[0][0]
        
        if prediction_proba > 0.5:
            st.error(f"**Status: AWAS BANJIR**")
            st.metric(label="Tingkat Keyakinan Model", value=f"{prediction_proba:.0%}")
        else:
            st.success(f"**Status: AMAN**")
            st.metric(label="Tingkat Keyakinan Model", value=f"{1-prediction_proba:.0%}")
    else:
        st.warning("Mohon unggah gambar terlebih dahulu untuk simulasi ketinggian air.")