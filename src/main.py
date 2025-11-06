from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_and_evaluate_models, find_optimal_k
import os
import yaml
import json

def main():
    """
    Fungsi utama untuk menjalankan alur kerja prediksi diabetes.
    """
    # Muat konfigurasi
    with open('config/setting.yml', 'r') as file:
        config = yaml.safe_load(file)

    # Membuat direktori output jika belum ada
    if not os.path.exists('output'):
        os.makedirs('output')

    # Muat dan proses data
    file_path = 'dataset/diabetes.csv'
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(
        data,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    # Temukan nilai k yang optimal untuk k-NN
    print("--- Mencari Nilai K Optimal untuk k-NN ---")
    optimal_k = find_optimal_k(
        X_train, y_train, X_test, y_test,
        output_path='output/knn_optimal_k_search.png'
    )
    config['models']['K-Nearest Neighbors']['n_neighbors'] = optimal_k
    print("\n")

    # Latih dan evaluasi model
    results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, config['models']
    )

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

    # Simpan hasil numerik ke file JSON
    json_results = {}
    for name, metrics in results.items():
        json_results[name] = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "confusion_matrix": metrics["confusion_matrix"].tolist()
        }

    with open('output/evaluation_metrics.json', 'w') as f:
        json.dump(json_results, f, indent=4)
    
    print("Hasil evaluasi numerik disimpan di output/evaluation_metrics.json")

if __name__ == "__main__":
    main()