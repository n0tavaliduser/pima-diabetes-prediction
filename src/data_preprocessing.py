import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_and_preprocess_data(file_path):
    """
    Memuat, membersihkan, dan menskalakan data.
    """
    df = pd.read_csv(file_path)
    
    # Ganti nilai 0 dengan NaN
    columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[columns_to_replace] = df[columns_to_replace].replace(0, np.nan)

    # Isi nilai NaN dengan median
    for col in columns_to_replace:
        df[col].fillna(df[col].median(), inplace=True)

    # Pisahkan fitur dan target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Penskalaan fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y