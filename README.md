# Klasifikasi-Penyakit-Jantung <3

Proyek ini dibuat untuk membantu mengklasifikasikan apakah seseorang **berisiko terkena penyakit jantung** atau tidak, berdasarkan data medis.  
Dengan memanfaatkan **Machine Learning**, model ini bisa menjadi langkah awal untuk mendukung analisis kesehatan.


### 🎯 Tujuan
- Membuat model klasifikasi sederhana untuk mendeteksi penyakit jantung.  
- Memberikan gambaran bagaimana *machine learning* bisa digunakan dalam bidang kesehatan.  
- Menyediakan contoh implementasi yang mudah dijalankan untuk pembelajaran.


### 📦 Apa yang Ada di Repo Ini?

| File / Folder | Isi |
|---------------|-----|
| `heart.csv`   | Dataset pasien (fitur medis & label penyakit jantung) |
| `main.py`     | Script utama: load data → training → evaluasi model |
| `README.md`   | Dokumentasi (file ini) |

---


### ⚙️ Instalasi
Sebelum menjalankan project, pastikan kamu sudah install Python 3.x dan beberapa library berikut:


```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 🚀 Cara Menjalankan
1. Clone repositori ini:

```bash
git clone https://github.com/SyaiYesMom/Klasifikasi-Penyakit-Jantung.git
cd Klasifikasi-Penyakit-Jantung
```

2. Jalankan script utama:

```bash
python main.py
```

3. Hasil yang akan keluar biasanya berupa:
- Akurasi model
- Confusion Matrix
- Prediksi apakah pasien berisiko atau tidak
- Perbandingan pasien yang terkena penyakit jantung atau tidak nya

### 🧪 Contoh Output
1. 5 Dataset teratas dari Dataset Penyakit Jantung

<img width="860" height="270" alt="Screenshot 2025-09-25 143334" src="https://github.com/user-attachments/assets/3074b605-7467-4b5c-8032-15e27dfb47b4" />

2. Output Machine Learning dengan model Logistic Regression

<img width="1744" height="996" alt="Screenshot 2025-09-25 143412" src="https://github.com/user-attachments/assets/ddae7678-7e69-4415-b28d-5f8b4fedaae9" />

3. Output Machine Learning dengan model Decision Tree
<img width="1745" height="1001" alt="Screenshot 2025-09-25 143452" src="https://github.com/user-attachments/assets/d9483366-09f4-4f29-b13d-f23f6a86f65d" />

4. Output Machine Learning dengan model SVM

<img width="1748" height="997" alt="Screenshot 2025-09-25 143523" src="https://github.com/user-attachments/assets/b326be43-03e8-4674-8be5-ec241650a2c7" />
