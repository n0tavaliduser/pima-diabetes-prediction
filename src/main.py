from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_and_evaluate_models
import os

def main():
    """
    Fungsi utama untuk menjalankan alur kerja prediksi diabetes.
    """
    # Membuat direktori output jika belum ada
    if not os.path.exists('output'):
        os.makedirs('output')

    # Muat dan proses data
    file_path = 'dataset/diabetes.csv'
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Latih dan evaluasi model
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Cetak hasil
    for name, metrics in results.items():
        print(f"--- {name} ---")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1_score']:.4f}")
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])
        print("\n")

if __name__ == "__main__":
    main()