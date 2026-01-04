# Analisis Dampak Faktor Cuaca Terhadap Produksi Padi Melalui Klasterisasi K-Means

Proyek sains data ini bertujuan untuk menganalisis dinamika produksi tanaman pangan di Pulau Sumatra (periode 2018–2024) melalui pendekatan *data-driven*. Fokus utama proyek ini adalah mengidentifikasi dampak perubahan iklim terhadap produktivitas padi menggunakan algoritma **K-Means Clustering** dan analisis regresi.

## 👥 Anggota Tim (Kelompok 3)
* **Abrar Wujedaan** [F1G123001]
* **Yayan Gisna Yudasti** [F1G123055]
* **Zacky Fiqran Kasmada** [F1G123038]

## 🎯 Tujuan & Pertanyaan Analisis
Proyek ini dirancang untuk menjawab pertanyaan bisnis berikut:
1.  Bagaimana hubungan curah hujan dan kelembapan dengan hasil panen pada setiap klaster wilayah?
2.  Bagaimana pengaruh variabel iklim (suhu, hujan, matahari, kelembapan) terhadap produktivitas (ku/ha)?
3.  Bagaimana interaksi antar indikator agroklimat terhadap tren produktivitas?
4.  Apakah hubungan luas panen dengan total volume produksi berbanding lurus secara konsisten?
5.  Bagaimana hubungan timbal balik antara indikator lingkungan dan kinerja pertanian?
6.  Seberapa besar peningkatan akurasi estimasi produktivitas jika menggunakan pendekatan zonasi iklim?

## 📂 Dataset
Data dimuat dari Google Drive dan terdiri dari tiga file utama:
1.  `data_iklim_sumatera.csv`: Berisi data Curah Hujan, Lama Penyinaran Matahari, dan Suhu Rata-rata.
2.  `data_produktivitas_sumatera.csv`: Berisi data produktivitas per provinsi (ku/ha).
3.  `Data_Tanaman_Padi_Sumatera.csv`: Dataset gabungan yang mencakup Produksi, Luas Panen, dan variabel cuaca.

## 🛠️ Teknologi & Pustaka
Proyek ini menggunakan **Python** di lingkungan Google Colab dengan pustaka berikut:
* **Pandas & Numpy**: Untuk manipulasi, pembersihan, dan analisis statistik data.
* **Matplotlib & Seaborn**: Untuk visualisasi data eksploratif.
* **Scikit-Learn**: Untuk pemodelan (*Linear Regression*, *Train-Test Split*).
* **Google Colab Drive**: Untuk integrasi penyimpanan awan.

## 📊 Alur Kerja (Pipeline)
Kode dalam notebook ini mengikuti tahapan berikut:

1.  **Environment Setup**: Mengimpor *library* dan menghubungkan Google Drive.
2.  **Data Wrangling**:
    * **Gathering**: Memuat dataset dan memvalidasi kelengkapan file.
    * **Inspection**: Melihat sampel data, tipe data, dan statistik deskriptif.
    * **Cleaning**: Mengecek duplikasi data dan menangani *outlier* (pencilan) menggunakan metode IQR (*Interquartile Range*).
3.  **Exploratory Data Analysis (EDA)**: Menganalisis tren dan pola data.
4.  **Insight Generation**: Menyusun rangkuman eksekutif berdasarkan tren data.

## 💡 Temuan Utama (Rangkuman Eksekutif)
Berdasarkan analisis awal data:
* **Tren Produksi**: Provinsi **Riau** dan **Bengkulu** mengalami kenaikan produksi signifikan, sedangkan **Lampung**, **Aceh**, dan **Sumsel** mengalami tren penurunan.
* **Efisiensi**: **Aceh** memiliki efisiensi lahan tertinggi. Kenaikan produksi di Riau lebih disebabkan oleh ekstensifikasi (luas lahan), bukan peningkatan produktivitas.
* **Faktor Cuaca**: Penyinaran matahari memiliki korelasi positif dengan hasil panen, sementara curah hujan yang terlalu tinggi berisiko menghambat produktivitas.
* **Ketergantungan**: Strategi pertanian di Sumatra masih sangat bergantung pada perluasan area tanam (*land-dependent*).

## 🚀 Cara Menjalankan
1.  Pastikan Anda memiliki akun Google dan akses ke Google Colab.
2.  Simpan dataset di folder Google Drive Anda: `/content/drive/MyDrive/dataset_ds/`.
3.  Jalankan setiap sel kode secara berurutan mulai dari *Import Libraries* hingga *Data Wrangling* dan *Analysis*.

---
*Dashboard Analisis Produksi Padi Sumatra 2018-2024*