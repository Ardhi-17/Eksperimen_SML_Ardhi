import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path_raw):
    """
    Memuat dataset dari file CSV.
    """
    return pd.read_csv(path_raw)


def handle_missing(df):
    """
    Menangani missing value pada kolom numerik dengan median.
    Ini dipilih karena median tidak sensitif terhadap outlier.
    """
    for col in df.select_dtypes(include='number').columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    return df


def remove_duplicates(df):
    """
    Menghapus baris duplikat untuk menjaga integritas data.
    """
    return df.drop_duplicates()


def encode_origin(df):
    """
    One-hot encoding untuk kolom kategorikal 'origin'.
    Ini menghindari urutan numerik palsu pada label encoding.
    """
    if 'origin' in df.columns:
        df = pd.get_dummies(df, columns=['origin'], drop_first=True)
    return df


def remove_outliers(df, column):
    """
    Menghapus outlier menggunakan metode IQR (Interquartile Range).
    Dipilih karena lebih robust dibanding z-score saat data tidak normal.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)
    return df[mask]


def scale_features(df):
    """
    Melakukan standarisasi pada kolom numerik.
    StandardScaler digunakan agar semua fitur numerik berada dalam skala yang seragam.
    """
    numeric_cols = df.select_dtypes(include='number').columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def preprocess_pipeline(path_in, path_out):
    """
    Pipeline preprocessing lengkap:
    1. Load dataset
    2. Bersihkan data
    3. Tangani outlier
    4. Encode kategorikal
    5. Standarisasi fitur
    6. Simpan hasilnya
    """
    df = load_data(path_in)
    df = handle_missing(df)
    df = remove_duplicates(df)
    df = encode_origin(df)

    # Tangani outlier pada kolom utama (jika ada)
    for col in ['mpg', 'horsepower']:
        if col in df.columns:
            df = remove_outliers(df, col)

    df = scale_features(df)

    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    df.to_csv(path_out, index=False)
    print(f"[✓] Preprocessed dataset saved at: {path_out}")

if __name__ == '__main__':
    # Lokasi file input/output
    root_dir = os.path.dirname(__file__)
    base_dir = os.path.abspath(os.path.join(root_dir, '..'))
    raw_csv = os.path.join(base_dir, 'namadataset_raw', 'Sleep_health_and_lifestyle_dataset.csv')
    output_csv = os.path.join(root_dir, 'namadataset_preprocessing', 'Sleep_health_and_lifestyle_dataset_preprocessed.csv')

    preprocess_pipeline(raw_csv, output_csv)
