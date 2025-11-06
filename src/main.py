from src.data_preprocessing import load_data, preprocess_data

def main():
    """
    Fungsi utama untuk menjalankan alur prediksi diabetes.
    """
    # Path ke dataset
    file_path = 'dataset/diabetes.csv'

    # Muat data
    df = load_data(file_path)

    # Pra-pemrosesan data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Cetak bentuk data untuk verifikasi
    print("Bentuk X_train:", X_train.shape)
    print("Bentuk X_test:", X_test.shape)
    print("Bentuk y_train:", y_train.shape)
    print("Bentuk y_test:", y_test.shape)

if __name__ == "__main__":
    main()