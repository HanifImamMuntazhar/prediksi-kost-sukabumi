import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Prediksi Kost Sukabumi",
    page_icon="🏠",
    layout="wide"
)

# 2. LOAD MODEL & FITUR AI
@st.cache_resource
def load_ai_model():
    model = joblib.load('model_rf_sukabumi.pkl')
    fitur_cols = joblib.load('kolom_fitur.pkl')
    return model, fitur_cols

model, fitur_cols = load_ai_model()

# 3. SIDEBAR NAVIGASI
st.sidebar.title("🏠 Navigasi")
menu = st.sidebar.radio(
    "Pilih Menu:",
    ("Hitung Prediksi Harga", "Insight & Fun Fact Data")
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Tentang Aplikasi:**\n"
    "Sistem ini ditenagai oleh Machine Learning (Random Forest) "
    "untuk memprediksi harga sewa kost di Sukabumi berdasarkan data riil."
)

# 4. HALAMAN 1: KALKULATOR PREDIKSI
if menu == "Hitung Prediksi Harga":
    st.title("💸 Kalkulator Harga Kost Sukabumi")
    st.markdown("Masukkan kriteria kost yang kamu inginkan, dan biarkan AI menebak harga sewanya!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Informasi Dasar")
        lokasi_pilihan = st.selectbox("Pilih Kecamatan (Lokasi):", [
            'Baros', 'Cibeureum', 'Cikole', 'Citamiang', 'Gunung Puyuh', 
            'Lembursitu', 'Warudoyong', 'Cisaat', 'Sukaraja', 'Selabintana'
        ])
        
        tipe_pilihan = st.selectbox("Pilih Tipe Kost:", ['Putra', 'Putri', 'Campur'])

    with col2:
        st.subheader("🛏️ Fasilitas Utama")
        st.write("Centang fasilitas yang tersedia:")
        
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            fas_ac = st.checkbox("AC")
            fas_wifi = st.checkbox("WiFi")
            fas_km_dalam = st.checkbox("Kamar Mandi Dalam")
        with f_col2:
            fas_kasur = st.checkbox("Kasur")
            fas_kloset = st.checkbox("Kloset Duduk")
            fas_24jam = st.checkbox("Akses 24 Jam")

    st.markdown("---")
    
    if st.button("🔮 Prediksi Harga Sekarang!", type="primary"):
        with st.spinner("AI sedang berhitung..."):
            input_data = {col: 0 for col in fitur_cols}
            
            nama_kolom_lokasi = f"Lokasi_{lokasi_pilihan}"
            nama_kolom_tipe = f"Tipe_Kost_{tipe_pilihan}"
            
            if nama_kolom_lokasi in input_data: input_data[nama_kolom_lokasi] = 1
            if nama_kolom_tipe in input_data: input_data[nama_kolom_tipe] = 1
            
            if fas_ac and "Fas_AC" in input_data: input_data["Fas_AC"] = 1
            if fas_wifi and "Fas_WiFi" in input_data: input_data["Fas_WiFi"] = 1
            if fas_km_dalam and "Fas_K_Mandi_Dalam" in input_data: input_data["Fas_K_Mandi_Dalam"] = 1
            if fas_kasur and "Fas_Kasur" in input_data: input_data["Fas_Kasur"] = 1
            if fas_kloset and "Fas_Kloset_Duduk" in input_data: input_data["Fas_Kloset_Duduk"] = 1
            if fas_24jam and "Fas_Akses_24_Jam" in input_data: input_data["Fas_Akses_24_Jam"] = 1

            df_input = pd.DataFrame([input_data])
            
            prediksi = model.predict(df_input)[0]
            
            st.success("Tebakan Selesai!")
            st.metric(label="Estimasi Harga Sewa (Per Bulan)", value=f"Rp {int(prediksi):,}".replace(',', '.'))
            
            st.caption("*Catatan: Ini adalah prediksi AI berdasarkan data riil di lapangan. Harga asli bisa bervariasi bergantung negosiasi, luas kamar, atau umur bangunan.*")


# 5. HALAMAN 2: INSIGHT & FUN FACT
elif menu == "Insight & Fun Fact Data":
    st.title("📊 Insight Penelitian & Fun Fact")
    st.markdown("Penasaran apa yang sebenarnya membuat harga kost bisa sangat mahal atau murah? Berikut adalah temuan dari algoritma *Machine Learning*!")
    
    try:
        img = Image.open('grafik_feature_importance.png')
        st.image(img, caption="Hasil Analisis Feature Importance Random Forest", use_column_width=True)
    except FileNotFoundError:
        st.warning("⚠️ Gambar grafik_feature_importance.png tidak ditemukan di folder. Pastikan file gambar sudah ada.")

    st.markdown("### 💡 3 Fun Fact Menarik Seputar Kost di Sukabumi")
    
    st.info("**1. AC Adalah Raja Harga (Pengaruh 57.47%)**\n\n"
            "Ternyata, ketersediaan AC menjadi garis batas mutlak antara kost standar dan kost premium. Algoritma menemukan bahwa pemilik kost menaikkan harga paling drastis hanya dengan menambahkan fasilitas AC, jauh mengalahkan pengaruh fasilitas lain.")
    
    st.success("**2. Cibeureum Punya 'Harga Spesial' (Pengaruh 8.89%)**\n\n"
               "Dari 10 wilayah yang diteliti, algoritma memetakan bahwa properti di area Cibeureum memiliki segmentasi pasar dan pola harga yang paling khas dibandingkan pusat kota. Letak geografis ini menjadi patokan penting kedua bagi mesin.")
    
    st.warning("**3. Aturan Gender Tidak Memengaruhi Harga (Pengaruh < 1%)**\n\n"
               "Ada mitos bahwa kost khusus putri lebih mahal karena fasilitas keamanan ekstra. Namun, data membuktikan hal itu salah! Secara empiris, algoritma melihat bahwa tipe kost (Putra/Putri/Campur) hampir tidak memberikan pengaruh sama sekali terhadap fluktuasi harga.")
