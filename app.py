import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Load model dan scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Sidebar navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Informasi", "Prediksi", "Visualisasi"])

# ===== HALAMAN BERANDA =====
if page == "Beranda":
    st.title("Selamat Datang di Aplikasi Deteksi Dini Siswa Korban Perundungan")
    st.write("""
    Aplikasi ini bertujuan untuk mendeteksi dini siswa yang berisiko menjadi korban perundungan dengan menggunakan algoritma 
    Support Vector Machine (SVM). Input yang diberikan berupa data dari kuesioner yang diisi oleh siswa.
    Pilih menu di sebelah kiri untuk mencoba prediksi atau melihat visualisasi dan informasi lebih lanjut.
    """)

# ===== HALAMAN INFORMASI =====
elif page == "Informasi":
    st.title("Apa itu Perundungan?")
    st.write("""
    Perundungan (**bullying**) adalah bentuk kekerasan yang dilakukan secara berulang, bisa dalam bentuk fisik, verbal, sosial, 
    maupun digital (cyberbullying), terhadap seseorang dengan tujuan merendahkan atau mengintimidasi. Perundungan dapat terjadi 
    di sekolah, lingkungan sekitar, atau melalui media daring. Dampak perundungan terhadap korban sangat serius, termasuk 
    gangguan psikologis, rendahnya harga diri, hingga keinginan bunuh diri.
    """)

    st.markdown("## 🔍 Bentuk-Bentuk Perundungan")
    st.markdown("""
    **Perundungan Fisik**
    - Memukul, menendang, mendorong.
    - Merusak barang milik korban.

    **Perundungan Verbal**
    - Menghina, mengejek, memanggil dengan nama julukan.
    - Mengancam.

    **Perundungan Sosial**
    - Mengucilkan dari kelompok.
    - Menyebarkan gosip atau fitnah.

    **Perundungan Daring (Cyberbullying)**
    - Mengirim pesan kebencian melalui media sosial.
    - Menyebarkan foto atau informasi memalukan.
    """)

    st.markdown("## ⚠️ Dampak Perundungan")
    st.write("""
    Perundungan bukan hanya masalah sepele. Dampaknya dapat bersifat jangka panjang, antara lain:
    - **Psikologis:** cemas, depresi, trauma.
    - **Sosial:** menarik diri, sulit percaya orang lain.
    - **Akademik:** prestasi sekolah menurun.
    - **Fisik:** gangguan tidur, sakit kepala, nyeri kronis.
    - **Ekstrem:** muncul pikiran untuk bunuh diri.
    """)

    st.markdown("## 🎯 Apa yang Menyebabkan Perundungan Terjadi?")
    st.markdown("""
    ✅ **Perbedaan Fisik & Identitas**  
    Misalnya penampilan, ras, agama, orientasi seksual.

    ✅ **Kurangnya Empati & Kontrol Diri**  
    Tidak bisa memahami perasaan orang lain atau mudah marah.

    ✅ **Lingkungan yang Mendukung Kekerasan**  
    Sekolah atau rumah yang membiarkan ejekan dan intimidasi.

    ✅ **Tekanan Teman Sebaya**  
    Ingin diterima kelompok, jadi ikut-ikutan mengejek.

    ✅ **Pengawasan Orang Dewasa yang Lemah**  
    Tidak ada pendampingan guru/orang tua.

    ✅ **Penyalahgunaan Media Sosial**  
    Menyebarkan hinaan atau ancaman secara online.

    """)

    st.markdown("## 🛡️ Cara Mencegah dan Mengatasi Perundungan")
    st.write("""
    - **Edukasi:** ajarkan empati, hormat, dan keberagaman.
    - **Pelaporan:** sediakan mekanisme mudah untuk melapor.
    - **Pendampingan:** dukungan psikolog bagi korban.
    - **Kebijakan Sekolah:** aturan tegas melawan perundungan.
    - **Keterlibatan Orang Tua:** komunikasi terbuka dan pengawasan.
    - **Literasi Digital:** ajarkan etika berinternet.
    
    ✨ **Ingat!**  
    Perundungan bukan salah korban. Semua pihak perlu ikut mencegah agar lingkungan jadi lebih aman dan menghargai perbedaan.
    """)

    st.markdown("## 📚 Referensi Jurnal Ilmiah")
    st.markdown("""
    - Smith, P. K., & Brain, P. (2000). *Bullying in schools: Lessons from two decades of research*. Aggressive Behavior, 26(1), 1–9.  
      [Link](https://doi.org/10.1002/(SICI)1098-2337(2000)26:1<1::AID-AB1>3.0.CO;2-7)
    - Kowalski, R. M., Giumetti, G. W., Schroeder, A. N., & Lattanner, M. R. (2014). *Bullying in the digital age: A critical review and meta-analysis of cyberbullying research among youth*. Psychological Bulletin, 140(4), 1073–1137.  
      [Link](https://doi.org/10.1037/a0035618)
    - Olweus, D. (1993). *Bullying at School: What We Know and What We Can Do*. Blackwell Publishers.
    - Hinduja, S., & Patchin, J. W. (2008). *Cyberbullying: An exploratory analysis of factors related to offending and victimization*. Deviant Behavior, 29(2), 129–156.  
      [Link](https://doi.org/10.1080/01639620701457816)
    """)

# ===== HALAMAN PREDIKSI DATA BARU =====
elif page == "Prediksi":
    st.title("Prediksi Deteksi Dini Siswa Korban Perundungan")

    st.write("""
    Silakan unggah file Excel berisi jawaban kuesioner siswa (Q1 - Q27) beserta kolom Nama, Kelas, dan Jenis Kelamin.
    Pastikan format kolom sama persis seperti data pelatihan.
    """)

    uploaded_file = st.file_uploader("Unggah file Excel", type=["xlsx"])

    if uploaded_file is not None:
        try:
            # Baca file
            data = pd.read_excel(uploaded_file)
            st.write("Data yang diunggah:")
            st.write(data.head())

            # Pastikan kolom wajib ada
            required_cols = ["Nama", "Kelas", "Jenis kelamin"]
            missing_required = [col for col in required_cols if col not in data.columns]
            if missing_required:
                st.error(f"Kolom berikut tidak ditemukan di file: {missing_required}")
            else:
                # Pastikan semua Q1 - Q27 ada
                question_columns = [f"Q{i}" for i in range(1,28)]
                missing_q = [col for col in question_columns if col not in data.columns]
                if missing_q:
                    st.error(f"Kolom berikut tidak ditemukan di file: {missing_q}")
                else:
                    # Ambil fitur
                    data_features = data[question_columns]

                    # Skala data
                    scaled_data = scaler.transform(data_features)

                    # Prediksi
                    predictions = model.predict(scaled_data)

                    # Tambahkan kolom hasil prediksi
                    data['Status'] = np.where(predictions == 1, "Siswa beresiko menjadi korban perundungan", "Siswa bukan Korban")

                    # Tampilkan hasil
                    st.write("Hasil Prediksi:")
                    st.write(data[['Nama', 'Kelas', 'Jenis kelamin', 'Status']])

                    # Grafik distribusi hasil prediksi
                    st.write("Distribusi Hasil Prediksi:")
                    pred_counts = data['Status'].value_counts()
                    fig2, ax2 = plt.subplots()
                    ax2.pie(
                        pred_counts,
                        labels=pred_counts.index,
                        autopct="%1.1f%%",
                        startangle=90,
                        colors=["#66b3ff", "#ff9999"]
                    )
                    ax2.axis("equal")
                    st.pyplot(fig2)

                    # Tampilkan jumlah siswa per kategori di bawah pie chart
                    jumlah_korban = pred_counts.get("Siswa beresiko menjadi korban perundungan", 0)
                    jumlah_bukan = pred_counts.get("Siswa bukan Korban", 0)

                    st.write(
                        f"**Jumlah siswa beresiko menjadi korban perundungan:** {jumlah_korban} | "
                        f"**Jumlah siswa bukan korban:** {jumlah_bukan}"
                    )

                    # Simpan gambar ke buffer
                    buf = io.BytesIO()
                    fig2.savefig(buf, format="png")
                    buf.seek(0)

                    # Tombol download gambar
                    st.download_button(
                        label="Download Diagram Pie",
                        data=buf,
                        file_name="distribusi_prediksi.png",
                        mime="image/png"
                    )

                    # Download hasil prediksi
                    csv = data.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download hasil prediksi CSV",
                        csv,
                        "hasil_prediksi.csv",
                        "text/csv"
                    )

        except Exception as e:
            st.error(f"Terjadi error saat memproses file: {e}")

# ===== HALAMAN VISUALISASI =====
elif page == "Visualisasi":
    st.title("Visualisasi Evaluasi Model Deteksi Perundungan")

    st.write("""
    Halaman ini menampilkan visualisasi hasil evaluasi model Support Vector Machine
    yang telah dilatih untuk mendeteksi siswa berisiko menjadi korban perundungan.
    """)

    # Load y_test dan y_pred
    y_test = np.load("y_test.npy")
    y_pred = np.load("y_pred.npy")

    # Tampilkan Classification Report
    st.markdown("### Classification Report")
    st.markdown("""
    |               | precision | recall | f1-score | support |
    |---------------|-----------|--------|----------|---------|
    | **-1**         | 1.00      | 0.98   | 0.99     | 52      |
    | **1**         | 0.86      | 1.00   | 0.92     | 6       |
    | **accuracy**  |           |        | 0.98     | 58      |
    | **macro avg** | 0.93      | 0.99   | 0.96     | 58      |
    | **weighted avg** | 0.99   | 0.98   | 0.98     | 58      |
    """)

    st.markdown("""
    **Cara Membaca:**  
    Tabel ini menunjukkan akurasi model mendeteksi siswa korban dan bukan korban. Nilai precision menunjukkan ketepatan prediksi, recall menunjukkan seberapa banyak kasus sebenarnya yang terdeteksi, dan f1-score adalah gabungan keduanya. Model memiliki akurasi keseluruhan 98% dengan f1-score kategori korban 0.92.
    """)

    # Tampilkan Confusion Matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Bukan Korban", "Korban"],
        yticklabels=["Bukan Korban", "Korban"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    st.pyplot(fig)

    st.markdown("""
    **Cara Membaca:**  
    Kotak ini menunjukkan jumlah prediksi yang benar dan salah. Kotak kiri atas menunjukkan 51 siswa yang memang bukan korban dan diprediksi benar, sedangkan kotak kanan bawah menunjukkan 6 siswa korban yang berhasil terdeteksi. Hanya 1 prediksi salah (bukan korban diprediksi korban).
    """)

    # Tampilkan distribusi label asli
    st.write("### Distribusi Label Asli di Data Uji")
    label_counts = pd.Series(y_test).value_counts().sort_index()
    fig2, ax2 = plt.subplots()
    ax2.bar(
        ["Bukan Korban", "Korban"],
        label_counts,
        color=["#66b3ff", "#ff9999"]
    )
    ax2.set_ylabel("Jumlah Siswa")
    st.pyplot(fig2)

    st.markdown("""
    **Cara Membaca:**  
    Grafik batang menunjukkan jumlah siswa pada data uji. Sebagian besar adalah bukan korban (52), dan sebagian kecil adalah korban (6). Model sudah dilatih agar tetap akurat meskipun datanya tidak seimbang.
    """)

    st.caption("Hasil evaluasi ini diperoleh dari dataset uji yang sudah diproses dan dilatih di Google Colab.")
