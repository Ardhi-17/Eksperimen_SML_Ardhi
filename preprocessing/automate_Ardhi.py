import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(path_raw):
    """Memuat dataset dari file CSV."""
    return pd.read_csv(path_raw)

def drop_unused_columns(df):
    """Menghapus kolom yang tidak relevan seperti ID unik."""
    cols_to_drop = ['Person ID']
    return df.drop(columns=cols_to_drop, errors='ignore')

def handle_missing(df):
    """Menangani missing value pada kolom numerik dengan median."""
    for col in df.select_dtypes(include='number').columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    return df

def remove_duplicates(df):
    """Menghapus baris duplikat."""
    return df.drop_duplicates()

def encode_categorical(df):
    """Label encode kolom kategorikal langsung pada kolom aslinya."""
    label_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
    for col in label_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))  # pastikan string
    return df

def clip_outliers(df):
    """Clipping outlier untuk kolom spesifik."""
    if 'Daily Steps' in df.columns:
        df['Daily Steps'] = df['Daily Steps'].clip(lower=2000, upper=11600)
    if 'Heart Rate' in df.columns:
        df['Heart Rate'] = df['Heart Rate'].clip(lower=62, upper=80)
    return df

def scale_features(df, target_column='Sleep Disorder'):
    """Standarisasi kolom numerik KECUALI target."""
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def preprocess_pipeline(path_in, path_out):
    df = load_data(path_in)
    print("[INFO] Kolom awal:", df.columns.tolist())

    df = drop_unused_columns(df)
    df = handle_missing(df)
    df = remove_duplicates(df)

    # Gabungkan kategori Occupation minor ke 'Others'
    if 'Occupation' in df.columns:
        vc = df['Occupation'].value_counts()
        minority_cats = vc[vc < 10].index
        df['Occupation'] = df['Occupation'].apply(lambda x: x if x not in minority_cats else 'Others')

    if 'Sleep Disorder' in df.columns:
        df['Sleep Disorder'] = df['Sleep Disorder'].replace(
            to_replace=[None, 'none', 'NONE', '', np.nan, pd.NA, 'nan'],
            value='None'
        )
        df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
        print("[DEBUG] Sleep Disorder value counts setelah standar:", df['Sleep Disorder'].value_counts())

    df = encode_categorical(df)
    df = clip_outliers(df)
    df = scale_features(df)

    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    df.to_csv(path_out, index=False)

    print(f"[✓] Preprocessed dataset saved at: {path_out}")
    print("[INFO] Kolom akhir:", df.columns.tolist())

if __name__ == '__main__':
    root_dir = os.path.dirname(__file__)
    raw_csv = os.path.abspath(os.path.join(root_dir, '..', 'namadataset_raw', 'Sleep_health_and_lifestyle_dataset.csv'))
    output_csv = os.path.join(root_dir, 'namadataset_preprocessing', 'Sleep_health_and_lifestyle_dataset_preprocessed.csv')

    preprocess_pipeline(raw_csv, output_csv)
