import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# === Parsing CLI argument ===
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

# === Standardisasi fitur numerik ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Simpan scaler ===
scaler_path = os.path.join(args.output, 'scaler_sleep.joblib')
joblib.dump(scaler, scaler_path)

# === Class weight untuk data imbang ===
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
cw_dict = dict(zip(np.unique(y_train), class_weights))

# === Konversi ke array (untuk kompatibilitas dengan mlflow) ===
y_train = np.array(y_train)

# === Start MLflow run ===
with mlflow.start_run():
    # Latih model
    clf = RandomForestClassifier(random_state=42, class_weight=cw_dict)
    clf.fit(X_train_scaled, y_train)

    # Log model ke MLflow Tracking (dengan input_example)
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="sleep-disorder-model",
        input_example=X_train_scaled[:1],
        registered_model_name=None
    )

    # Simpan model dalam format MLflow (untuk build Docker)
    mlflow_model_path = os.path.join(args.output, 'mlflow_model')
    mlflow.sklearn.save_model(
        sk_model=clf,
        path=mlflow_model_path,
        input_example=X_train_scaled[:1]
    )

    # Simpan juga sebagai joblib biasa
    model_path = os.path.join(args.output, 'model_sleep.joblib')
    joblib.dump(clf, model_path)

print(f"[✓] Model selesai dilatih dan disimpan ke: {args.output}")
