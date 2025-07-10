import streamlit as st
import pandas as pd
import joblib

# Load model pipeline yang sudah disimpan (SMOTE + SVM)
model = joblib.load('svm_model.pkl')

st.title("Deteksi Dini Korban Perundungan Siswa")

st.write("""
Upload file CSV yang berisi data kuesioner siswa (Q1 - Q27) untuk diprediksi.
""")

uploaded_file = st.file_uploader("Pilih file CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("**Data yang diupload:**")
    st.dataframe(data)

    # Drop kolom non-pertanyaan jika ada
    columns_to_drop = ['Nama', 'Kelas', 'Jenis Kelamin', 'skor_b', 'skor_f', 'skor_bf', 'label']
    data_for_prediction = data.drop(columns=[col for col in columns_to_drop if col in data.columns], errors="ignore")

    # Prediksi
    predictions = model.predict(data_for_prediction)

    # Tampilkan hasil
    data['Prediksi'] = predictions
    data['Status'] = data['Prediksi'].map({0: 'Bukan Korban', 1: 'Korban'})

    st.write("**Hasil Prediksi:**")
    st.dataframe(data[['Nama', 'Status']] if 'Nama' in data.columns else data[['Status']])

    # Download hasil
    csv = data.to_csv(index=False).encode()
    st.download_button(
        label="Download hasil prediksi CSV",
        data=csv,
        file_name='hasil_prediksi.csv',
        mime='text/csv'
    )
