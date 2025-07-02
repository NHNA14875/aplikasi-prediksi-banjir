import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from PIL import Image
from sklearn.preprocessing import StandardScaler
import easyocr
import re

# =============================================================================
# Bagian 0: Konfigurasi Halaman dan Gaya (CSS Cerdas)
# =============================================================================
def setup_page():
    """Mengatur konfigurasi halaman dan menyuntikkan CSS yang sadar tema."""
    st.set_page_config(
        page_title="Analisis Risiko Banjir",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # CSS yang menggunakan variabel tema Streamlit
    st.markdown("""
    <style>
        /* Mengubah font utama */
        html, body, [class*="st-"] {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        /* Header Utama */
        .main-header {
            font-size: 2.8rem;
            font-weight: 700;
            color: var(--primary-color);
            text-align: center;
            padding: 1rem 0;
        }
        /* Kontainer dengan border dan shadow yang beradaptasi */
        .st-emotion-cache-1r4qj8v {
            border: 1px solid var(--gray-80);
            border-radius: 10px;
            padding: 25px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            background-color: var(--secondary-background-color);
            transition: all 0.3s;
        }
        .st-emotion-cache-1r4qj8v:hover {
            box-shadow: 0 6px 16px rgba(0,0,0,0.12);
        }
        /* Tombol Utama */
        .stButton>button {
            border: none;
            border-radius: 10px;
            color: white;
            background: var(--primary-color);
            padding: 12px 24px;
            font-size: 1.2rem;
            font-weight: bold;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            filter: brightness(1.2);
            transform: translateY(-2px);
        }
        /* Hasil Prediksi yang beradaptasi */
        .result-safe { border-left: 8px solid #28a745; padding: 20px; border-radius: 8px; background-color: var(--secondary-background-color); }
        .result-warn { border-left: 8px solid #ffc107; padding: 20px; border-radius: 8px; background-color: var(--secondary-background-color); }
        .result-danger { border-left: 8px solid #dc3545; padding: 20px; border-radius: 8px; background-color: var(--secondary-background-color); }
        .result-safe h4, .result-warn h4, .result-danger h4 { color: var(--text-color); }
        .result-safe p, .result-warn p, .result-danger p { color: var(--text-color); }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# Bagian 1: Modul AI (Logika Tetap Sama)
# =============================================================================
@st.cache_resource
def train_and_get_model():
    n_samples = 2500
    np.random.seed(42)
    aman_hujan = np.random.uniform(0, 80, int(n_samples * 0.4))
    aman_air = np.random.uniform(50, 250, int(n_samples * 0.4))
    label_aman = np.full(int(n_samples * 0.4), 0.0)
    waspada_hujan_a = np.random.uniform(200, 400, int(n_samples * 0.15))
    waspada_air_a = np.random.uniform(150, 350, int(n_samples * 0.15))
    waspada_hujan_b = np.random.uniform(100, 200, int(n_samples * 0.15))
    waspada_air_b = np.random.uniform(350, 500, int(n_samples * 0.15))
    label_waspada = np.full(int(n_samples * 0.3), 0.5)
    banjir_hujan_a = np.random.uniform(150, 400, int(n_samples * 0.15))
    banjir_air_a = np.random.uniform(500, 750, int(n_samples * 0.15))
    banjir_hujan_b = np.random.uniform(10, 100, int(n_samples * 0.15))
    banjir_air_b = np.random.uniform(600, 800, int(n_samples * 0.15))
    label_banjir = np.full(int(n_samples * 0.3), 1.0)
    curah_hujan = np.concatenate([aman_hujan, waspada_hujan_a, waspada_hujan_b, banjir_hujan_a, banjir_hujan_b])
    tinggi_air = np.concatenate([aman_air, waspada_air_a, waspada_air_b, banjir_air_a, banjir_air_b])
    labels = np.concatenate([label_aman, label_waspada, label_banjir])
    df = pd.DataFrame({'curah_hujan_mm': curah_hujan, 'tinggi_muka_air_cm': tinggi_air, 'status': labels})
    df = df.sample(frac=1).reset_index(drop=True)
    X = df[['curah_hujan_mm', 'tinggi_muka_air_cm']].values
    y = df['status'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Sequential([
        Dense(16, activation='relu', input_shape=(2,)),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X_scaled, y, epochs=200, batch_size=64, verbose=0)
    return model, scaler

# =============================================================================
# Bagian 2: Modul Computer Vision (Logika Tetap Sama)
# =============================================================================
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en'])

def read_water_level_from_image(image_file, reader):
    try:
        pil_image = Image.open(image_file).convert('RGB')
        cv_image = np.array(pil_image)
        cv_image = cv_image[:, :, ::-1].copy()
        height, width, _ = cv_image.shape
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        mask[:height//2, :] = 0
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        y_water_line = height 
        if contours:
            c = max(contours, key=cv2.contourArea)
            y_water_line = cv2.boundingRect(c)[1]
        results = reader.readtext(cv_image)
        if not results:
            st.warning("OCR tidak menemukan teks/angka pada gambar. Menggunakan nilai default.")
            return 150, pil_image, "Tidak terbaca"
        best_match_value = None
        min_distance = float('inf')
        for (bbox, text, prob) in results:
            cleaned_text = re.sub(r'[^\d.-]', '', text)
            if not cleaned_text or cleaned_text == '.' or cleaned_text == '-': continue
            try:
                value = float(cleaned_text)
                (tl, tr, br, bl) = bbox
                y_center = (tl[1] + br[1]) / 2
                distance = abs(y_center - y_water_line)
                if distance < min_distance:
                    min_distance = distance
                    best_match_value = abs(int(value * 100)) if abs(value) < 10 else abs(int(value))
            except ValueError: continue
        if best_match_value is None:
            st.warning("OCR tidak dapat memvalidasi angka yang relevan. Menggunakan nilai default.")
            return 150, pil_image, "Tidak terbaca"
        return best_match_value, pil_image, f"{best_match_value} cm (terbaca)"
    except Exception as e:
        st.error(f"Terjadi kesalahan pada modul Computer Vision: {e}")
        return None, None, None

# =============================================================================
# Bagian 3: Tampilan Aplikasi (User Interface)
# =============================================================================
setup_page()

with st.sidebar:
    st.image("https://www.streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=150)
    st.title("Tentang Proyek")
    st.markdown("Aplikasi ini adalah implementasi dari proposal UAS **'Prediksi Banjir Berbasis Machine Learning'**.")
    st.markdown("---")
    st.subheader("Disusun oleh:")
    st.write("**Yobby Azriel Iqdhi Vianta**")
    st.write("**A11.2023.14890**")
    st.write("**A11.4408**")
    st.markdown("---")
    st.write("Dosen Pengampu:")
    st.write("**Dr. Ricardus Anggi P.**")
    st.markdown("---")
    st.info("Aplikasi v6.0 (Koreksi OCR)")

st.markdown('<p class="main-header">Sistem Peringatan Dini Banjir</p>', unsafe_allow_html=True)

with st.spinner("Mempersiapkan model AI prediksi..."):
    model, scaler = train_and_get_model()

col1, col2 = st.columns((1, 1), gap="large")

with col1:
    with st.container():
        st.subheader("1. Input Data Lingkungan")
        st.markdown("##### 🌧️ Curah Hujan (mm)")
        curah_hujan = st.slider("Geser untuk mengatur perkiraan curah hujan harian.", 0, 400, 150, label_visibility="collapsed")
        
        st.markdown("##### 📸 Ketinggian Muka Air (cm)")
        uploaded_file = st.file_uploader("Unggah foto papan duga air...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        # Inisialisasi variabel di session state
        if 'tinggi_air_final' not in st.session_state:
            st.session_state.tinggi_air_final = 0
            
        if uploaded_file:
            with st.spinner("Menganalisis gambar dengan OCR..."):
                reader = get_ocr_reader()
                level, image, status_text = read_water_level_from_image(uploaded_file, reader)
            
            if level is not None and image is not None:
                st.image(image, caption=f"Gambar diunggah.", use_column_width=True)
                st.write(f"**Ketinggian Air Terbaca (OCR):** `{status_text}`")
                
                # FITUR BARU: Opsi untuk Koreksi Manual
                koreksi = st.checkbox("✔️ Koreksi hasil OCR?")
                if koreksi:
                    ketinggian_manual = st.number_input("Masukkan ketinggian air yang benar (cm):", min_value=0, value=level)
                    st.session_state.tinggi_air_final = ketinggian_manual
                else:
                    st.session_state.tinggi_air_final = level
            else:
                st.session_state.tinggi_air_final = 0
        else:
            st.session_state.tinggi_air_final = 0

with col2:
    with st.container():
        st.subheader("2. Hasil Analisis Risiko")
        result_placeholder = st.empty()
        result_placeholder.info("Hasil analisis risiko akan ditampilkan di sini setelah Anda menekan tombol 'Analisis'.")

st.markdown("---")
if st.button("🚀 Analisis Risiko Sekarang", use_container_width=True):
    # Menggunakan nilai final dari session_state
    if st.session_state.tinggi_air_final > 0:
        input_data = np.array([[curah_hujan, st.session_state.tinggi_air_final]])
        input_data_scaled = scaler.transform(input_data)
        prediction_score = model.predict(input_data_scaled)[0][0]
        
        with result_placeholder.container():
            if prediction_score > 0.75:
                st.markdown(f'<div class="result-danger"><h4>Status: AWAS BANJIR 🚨</h4><p>Kombinasi curah hujan dan tinggi muka air sangat berbahaya. Segera lakukan evakuasi.</p><strong>Indeks Risiko: {prediction_score:.2f}</strong></div>', unsafe_allow_html=True)
            elif prediction_score > 0.4:
                st.markdown(f'<div class="result-warn"><h4>Status: WASPADA ⚠️</h4><p>Kondisi berpotensi menjadi berbahaya. Siapkan langkah-langkah mitigasi.</p><strong>Indeks Risiko: {prediction_score:.2f}</strong></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-safe"><h4>Status: AMAN ✅</h4><p>Kondisi saat ini terpantau aman.</p><strong>Indeks Risiko: {prediction_score:.2f}</strong></div>', unsafe_allow_html=True)
    else:
        st.warning("Mohon unggah gambar terlebih dahulu untuk analisis ketinggian air.")