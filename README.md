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
| K-Nearest Neighbors   | 0.740   | 0.659   | 0.537  | 0.592    |
| Decision Tree         | 0.682   | 0.553   | 0.481  | 0.515    |
| Naive Bayes           | 0.701   | 0.567   | 0.630  | 0.596    |

### Analisis Hasil

-   **K-Nearest Neighbors (KNN)**: Model ini mencapai akurasi tertinggi (74.0%) di antara ketiganya. Presisinya juga cukup baik (65.9%), yang berarti dari semua prediksi positif yang dibuat, 65.9% di antaranya benar. Namun, recall-nya (53.7%) lebih rendah, yang menunjukkan bahwa model ini melewatkan sejumlah kasus diabetes positif.

-   **Decision Tree**: Model ini memiliki performa terendah di antara ketiganya di hampir semua metrik. Akurasinya adalah 68.2%, dan F1-score-nya adalah 51.5%. Ini menunjukkan bahwa model Decision Tree mungkin kurang cocok untuk dataset ini dibandingkan dengan model lainnya.

-   **Naive Bayes**: Model ini memiliki F1-score tertinggi (59.6%) dan recall tertinggi (63.0%), yang berarti model ini paling baik dalam mengidentifikasi kasus diabetes positif. Meskipun akurasinya sedikit lebih rendah dari KNN, kemampuannya untuk meminimalkan false negatives (kasus yang salah diprediksi sebagai non-diabetes) menjadikannya model yang kuat untuk kasus penggunaan ini.

Secara keseluruhan, **K-Nearest Neighbors** memberikan akurasi tertinggi, tetapi **Naive Bayes** lebih unggul dalam hal F1-score dan recall, yang mungkin lebih penting dalam konteks medis di mana melewatkan kasus positif (false negative) lebih berisiko daripada salah mengidentifikasi kasus negatif (false positive).