import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(file_path):
    """
    Memuat data dari file CSV.

    Parameters:
    file_path (str): Path ke file CSV.

    Returns:
    pandas.DataFrame: DataFrame yang berisi data.
    """
    return pd.read_csv(file_path)

def preprocess_data(df, test_size, random_state):
    """
    Melakukan pra-pemrosesan data. Ini termasuk menangani nilai nol yang tidak masuk akal
    dan memisahkan fitur dan target.

    Parameters:
    df (pandas.DataFrame): DataFrame input.
    test_size (float): Proporsi dataset untuk disertakan dalam pemisahan pengujian.
    random_state (int): Mengontrol pengacakan yang diterapkan pada data.

    Returns:
    tuple: Tuple yang berisi X_train, X_test, y_train, y_test.
    """
    # Ganti nilai 0 dengan NaN pada kolom-kolom tertentu di mana 0 tidak masuk akal
    columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[columns_to_replace] = df[columns_to_replace].replace(0, np.nan)

    # Isi nilai NaN dengan median dari setiap kolom
    for col in columns_to_replace:
        df[col].fillna(df[col].median(), inplace=True)

    # Pisahkan fitur (X) dan target (y)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Pisahkan data menjadi set pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Penskalaan fitur
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test