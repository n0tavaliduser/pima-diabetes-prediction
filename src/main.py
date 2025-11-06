from src.data_preprocessing import load_data
from src.model_training import run_experiments
import os
import yaml
import json
import pandas as pd

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

    # Muat data
    file_path = 'dataset/diabetes.csv'
    data = load_data(file_path)
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Jalankan eksperimen
    results = run_experiments(X, y, config['experiments'])

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