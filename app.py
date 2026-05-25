import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

@st.cache_resource
def load_artifacts():
    with open('model_gizi.pkl',   'rb') as f: model   = pickle.load(f)
    with open('scaler_gizi.pkl',  'rb') as f: scaler  = pickle.load(f)
    with open('encoder_gizi.pkl', 'rb') as f: encoder = pickle.load(f)
    return model, scaler, encoder

model, scaler, encoder = load_artifacts()

st.set_page_config(page_title="Klasifikasi Status Gizi", page_icon="🥗", layout="wide")
st.title("🥗 Sistem Klasifikasi Status Gizi")
st.markdown("**Program Makan Bergizi Gratis — Universitas AMIKOM Yogyakarta**")
st.divider()

st.sidebar.header("📋 Input Data ")
umur           = st.sidebar.slider("Umur", 0, 60, 24)
jenis_kelamin  = st.sidebar.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
berat_badan    = st.sidebar.number_input("Berat Badan (kg)", 1.0, 80.0, 12.0, step=0.1)
tinggi_badan   = st.sidebar.number_input("Tinggi Badan (cm)", 40.0, 180.0, 85.0, step=0.1)
asupan_kalori  = st.sidebar.number_input("Asupan Kalori (kkal)", 0.0, 2000.0, 200.0, step=1.0)
protein        = st.sidebar.number_input("Protein (gram)", 0.0, 100.0, 10.0, step=0.1)
frekuensi_mbg  = st.sidebar.slider("Frekuensi MBG (kali/minggu)", 1, 5, 3)
kehadiran      = st.sidebar.slider("Kehadiran (%)", 0.0, 100.0, 80.0, step=0.5)
nilai_akademik = st.sidebar.slider("Nilai Akademik", 0.0, 100.0, 75.0, step=0.5)

bmi = round(berat_badan / (tinggi_badan / 100) ** 2, 2)
st.sidebar.metric("BMI (otomatis)", bmi)
jk_encoded = 0 if jenis_kelamin == "Laki-laki" else 1
prediksi_btn = st.sidebar.button("🔍 Prediksi Status Gizi", use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Data yang Diinput")
    input_dict = {
        "Umur"                : str(umur),
        "Jenis Kelamin"       : jenis_kelamin,
        "Berat Badan (kg)"    : str(berat_badan),
        "Tinggi Badan (cm)"   : str(tinggi_badan),
        "BMI"                 : str(bmi),
        "Asupan Kalori (kkal)": str(asupan_kalori),
        "Protein (gram)"      : str(protein),
        "Frekuensi MBG"       : str(frekuensi_mbg),
        "Kehadiran (%)"       : str(kehadiran),
        "Nilai Akademik"      : str(nilai_akademik),
    }
    st.dataframe(
        pd.DataFrame(input_dict.items(), columns=["Variabel", "Nilai"]),
        hide_index=True
    )

with col2:
    st.subheader("🎯 Hasil Prediksi")
    if prediksi_btn:
        input_array = np.array([[
            umur, jk_encoded, berat_badan, tinggi_badan, bmi,
            asupan_kalori, protein, frekuensi_mbg, kehadiran, nilai_akademik
        ]])
        input_scaled = scaler.transform(input_array)
        pred_encoded = model.predict(input_scaled)[0]
        pred_proba   = model.predict_proba(input_scaled)[0]
        pred_label   = encoder.inverse_transform([pred_encoded])[0]

        warna = {"normal": "🟢", "tinggi": "🔵", "stunted": "🟡", "severely stunted": "🔴"}
        ikon  = warna.get(pred_label, "⚪")
        st.success(f"### {ikon} Status Gizi: **{pred_label.upper()}**")

        proba_df = pd.DataFrame({
            "Status Gizi"     : encoder.classes_,
            "Probabilitas (%)": (pred_proba * 100).round(2)
        }).sort_values("Probabilitas (%)", ascending=False)

        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ['#4CAF50' if c == pred_label else '#B0BEC5' for c in proba_df['Status Gizi']]
        ax.barh(proba_df['Status Gizi'], proba_df['Probabilitas (%)'], color=colors)
        ax.set_xlabel("Probabilitas (%)")
        ax.set_xlim(0, 100)
        for i, v in enumerate(proba_df['Probabilitas (%)']):
            ax.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("← Isi data di sidebar lalu klik Prediksi Status Gizi")

st.divider()
st.subheader("📂 Prediksi Batch (Upload CSV)")
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    df_upload = pd.read_csv(uploaded_file)
    st.write("Preview data:", df_upload.head())
    fitur = ['Umur','Jenis_Kelamin','Berat_Badan','Tinggi_Badan','BMI',
             'Asupan_Kalori','Protein','Frekuensi_MBG','Kehadiran','Nilai_Akademik']
    if all(col in df_upload.columns for col in fitur):
        X_batch = scaler.transform(df_upload[fitur])
        y_batch = model.predict(X_batch)
        df_upload['Prediksi_Status_Gizi'] = encoder.inverse_transform(y_batch)
        st.success(f"Prediksi selesai untuk {len(df_upload):,} data!")
        st.dataframe(df_upload.astype(str))
        csv_result = df_upload.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Hasil", csv_result, "hasil_prediksi.csv", "text/csv")
    else:
        st.error(f"Kolom yang dibutuhkan: {fitur}")

st.divider()
st.caption("Ginanjar Hilmi Yahya — 23.11.5800 | S1 Informatika UNIVERSITAS AMIKOM Yogyakarta 2025/2026")
