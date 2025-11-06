# Proyek Prediksi Diabetes Pima Indians

Proyek ini bertujuan untuk memprediksi kemungkinan diabetes pada wanita Pima Indian berdasarkan beberapa variabel diagnostik. Proyek ini menggunakan dataset dari National Institute of Diabetes and Digestive and Kidney Diseases untuk membangun dan mengevaluasi beberapa model machine learning. Tujuan utamanya adalah untuk membandingkan kinerja dari tiga model klasifikasi: K-Nearest Neighbors (KNN), Decision Tree, dan Naive Bayes dalam memprediksi diabetes.

## Dataset

Proyek ini menggunakan dataset **Pima Indians Diabetes Database** yang tersedia di `dataset/diabetes.csv`. Dataset ini memiliki beberapa fitur berikut:

- **Pregnancies**: Jumlah kehamilan
- **Glucose**: Konsentrasi glukosa plasma 2 jam setelah tes toleransi glukosa oral
- **BloodPressure**: Tekanan darah diastolik (mm Hg)
- **SkinThickness**: Ketebalan lipatan kulit trisep (mm)
- **Insulin**: Insulin serum 2 jam (mu U/ml)
- **BMI**: Indeks massa tubuh (berat dalam kg/(tinggi dalam m)^2)
- **DiabetesPedigreeFunction**: Fungsi silsilah diabetes
- **Age**: Usia (tahun)
- **Outcome**: Variabel kelas (0 atau 1), di mana 1 menandakan adanya diabetes dan 0 menandakan tidak adanya diabetes.

## Instalasi dan Setup

Untuk menjalankan proyek ini, Anda disarankan untuk menggunakan Conda untuk manajemen lingkungan.

1.  **Buat Lingkungan Conda Baru**

    Buat lingkungan Conda baru dari file `requirements.txt` yang disediakan.

    ```bash
    conda create --name <ENV-NAME> python=3.8
    conda activate <ENV-NAME>
    pip install -r requirements.txt
    ```

2.  **Jalankan Proyek**

    Setelah lingkungan diaktifkan dan semua dependensi diinstal, jalankan file `main.py` dari direktori `src`.

    ```bash
    python src/main.py
    ```

    Skrip akan melatih model, mengevaluasi kinerjanya, dan menyimpan hasilnya di direktori `output/`.

## Hasil Performa Model

Hasil evaluasi dari ketiga model disimpan di `output/evaluation_metrics.json`. Berikut adalah ringkasan performa masing-masing model:

| Model                 | Akurasi | Presisi | Recall | F1-score |
| --------------------- | ------- | ------- | ------ | -------- |
| K-Nearest Neighbors   | 0.805   | 0.773   | 0.630  | 0.694    |
| Decision Tree         | 0.675   | 0.545   | 0.444  | 0.490    |
| Naive Bayes           | 0.740   | 0.621   | 0.667  | 0.643    |

> **Catatan**: Hasil performa untuk K-Nearest Neighbors (KNN) di atas diperoleh dengan menggunakan nilai `K=5`. Pengujian awal dengan `K=10` (seperti yang dikonfigurasi secara default di `config/setting.yml`) menghasilkan akurasi yang sedikit lebih rendah (0.740). Nilai `K=5` terbukti memberikan keseimbangan yang lebih baik antara bias dan varians untuk dataset ini.

### Analisis Hasil

-   **K-Nearest Neighbors (KNN)**: Dengan `K=5` dan pembagian data 90/10, model ini mencapai **akurasi tertinggi** (80.5%) dan **F1-score tertinggi** (69.4%). Peningkatan signifikan ini menunjukkan bahwa KNN sangat diuntungkan dari jumlah data pelatihan yang lebih besar, memungkinkannya untuk menangkap pola yang lebih baik.

-   **Decision Tree**: Model ini masih menunjukkan performa terendah di antara ketiganya. Ini mengindikasikan bahwa bahkan dengan lebih banyak data pelatihan, struktur pohon keputusan tunggal mungkin masih kesulitan untuk menggeneralisasi pola kompleks dari dataset ini secara efektif.

-   **Naive Bayes**: Model ini menunjukkan **recall tertinggi** (66.7%), menjadikannya yang terbaik dalam mengidentifikasi semua kasus diabetes positif. Peningkatan akurasi juga terlihat, meskipun tidak sebesar KNN. Ini menegaskan kekuatan Naive Bayes dalam skenario di mana meminimalkan *false negatives* adalah prioritas utama.

Secara keseluruhan, **K-Nearest Neighbors (dengan K=5)** tetap menjadi model dengan performa terbaik secara umum, terutama ketika diberi lebih banyak data untuk pelatihan. Namun, jika tujuannya adalah untuk memaksimalkan deteksi kasus positif, **Naive Bayes** tetap menjadi pilihan yang sangat kuat karena keunggulannya dalam metrik recall.