import os
import argparse
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# === Argument CLI ===
parser = argparse.ArgumentParser(description="Train RandomForest for Sleep Disorder")
parser.add_argument('--input', type=str, required=True, help='Path to input CSV dataset')
parser.add_argument('--output', type=str, default="artifacts/sleep-disorder-model", help='Path to save model artifacts')
args = parser.parse_args()

# === Buat direktori output jika belum ada ===
os.makedirs(args.output, exist_ok=True)

# === Load dataset ===
df = pd.read_csv(args.input)

# === Fitur dan target ===
features = [
    'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
    'Physical Activity Level', 'Stress Level', 'BMI Category',
    'Heart Rate', 'Daily Steps'
]
target = 'Sleep Disorder'

X = df[features]
y = df[target]

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === Standarisasi fitur numerik ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Simpan scaler ===
scaler_path = os.path.join(args.output, 'scaler_sleep.joblib')
joblib.dump(scaler, scaler_path)

# === Hitung class weight ===
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
cw_dict = dict(zip(np.unique(y_train), class_weights))

# === MLflow run ===
with mlflow.start_run():
    clf = RandomForestClassifier(random_state=42, class_weight=cw_dict)
    clf.fit(X_train_scaled, y_train)

    # Simpan model
    model_path = os.path.join(args.output, 'model_sleep.joblib')
    joblib.dump(clf, model_path)

    print(f"[✓] Model disimpan ke: {model_path}")
    print(f"[✓] Scaler disimpan ke: {scaler_path}")
