from src.data_preprocessing import load_and_preprocess_data
from src.model_training import run_experiments
import os
import yaml
import json
import pandas as pd
import argparse

def main():
    """
    Fungsi utama untuk menjalankan alur kerja prediksi diabetes.
    """
    # Tambahkan argument parser untuk konfigurasi
    parser = argparse.ArgumentParser(description="Jalankan alur kerja prediksi diabetes.")
    parser.add_argument(
        '--config',
        type=str,
        default='config/setting.yml',
        help='Path ke file konfigurasi YAML.'
    )
    args = parser.parse_args()

    # Muat konfigurasi
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Membuat direktori output jika belum ada
    if not os.path.exists('output'):
        os.makedirs('output')

    # Muat dan proses data
    file_path = 'dataset/diabetes.csv'
    X_scaled, y = load_and_preprocess_data(file_path)

    # Jalankan eksperimen
    results = run_experiments(X_scaled, y, config['experiments'])

    # Simpan hasil ke file JSON
    with open('output/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Eksperimen selesai. Hasil disimpan di output/experiment_results.json")

    # Tampilkan hasil dalam format tabel
    print("\n--- Hasil Eksperimen ---")
    for validation_type, validation_results in results.items():
        print(f"\n{validation_type.replace('_', ' ').title()}:")
        for setting, model_results in validation_results.items():
            print(f"\n  Setting: {setting}")
            df = pd.DataFrame(model_results).T
            # Format a float in the dataframe to 4 decimal places
            for col in df.columns:
                if df[col].dtype == 'float64':
                    df[col] = df[col].map('{:.4f}'.format)
            print(df.to_string())

if __name__ == '__main__':
    main()