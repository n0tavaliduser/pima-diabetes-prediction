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
| K-Nearest Neighbors   | 0.753   | 0.660   | 0.611  | 0.635    |
| Decision Tree         | 0.682   | 0.553   | 0.481  | 0.515    |
| Naive Bayes           | 0.701   | 0.567   | 0.630  | 0.596    |

> **Catatan**: Hasil performa untuk K-Nearest Neighbors (KNN) di atas diperoleh dengan menggunakan nilai `K=5`. Pengujian awal dengan `K=10` (seperti yang dikonfigurasi secara default di `config/setting.yml`) menghasilkan akurasi yang sedikit lebih rendah (0.740). Nilai `K=5` terbukti memberikan keseimbangan yang lebih baik antara bias dan varians untuk dataset ini.

### Analisis Hasil

-   **K-Nearest Neighbors (KNN)**: Dengan `K=5`, model ini mencapai **akurasi dan F1-score tertinggi** (75.3% dan 63.5%) di antara ketiga model. Ini menunjukkan bahwa KNN adalah model yang paling seimbang dalam hal presisi dan recall, menjadikannya pilihan yang sangat baik secara keseluruhan.

-   **Decision Tree**: Model ini secara konsisten menunjukkan performa terendah di semua metrik utama, menunjukkan bahwa struktur pohon keputusan tunggal mungkin tidak cukup kompleks untuk menangkap pola dalam dataset ini tanpa overfitting.

-   **Naive Bayes**: Model ini unggul dalam **recall tertinggi** (63.0%), yang berarti paling andal dalam mengidentifikasi semua kasus diabetes positif. Namun, presisinya lebih rendah, yang berarti ada lebih banyak prediksi false positive dibandingkan KNN.

Secara keseluruhan, **K-Nearest Neighbors (dengan K=5)** adalah model dengan performa terbaik secara umum, menawarkan akurasi dan F1-score tertinggi. Namun, jika prioritas utamanya adalah untuk meminimalkan *false negatives* (tidak mendeteksi diabetes padahal sebenarnya ada), maka **Naive Bayes** bisa menjadi pilihan yang lebih baik karena recall-nya yang superior.