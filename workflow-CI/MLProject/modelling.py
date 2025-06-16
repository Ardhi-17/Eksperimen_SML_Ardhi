import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Aktifkan autolog dari MLflow
mlflow.sklearn.autolog()

# Load dataset
csv_path = os.path.join(os.path.dirname(__file__), 'Sleep_health_and_lifestyle_dataset_preprocessed.csv')
df = pd.read_csv(csv_path)

# Fitur dan target
features = [
    'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
    'Physical Activity Level', 'Stress Level', 'BMI Category',
    'Heart Rate', 'Daily Steps'
]
target = 'Sleep Disorder'

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Standardisasi fitur numerik
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hitung class weight
class_weights = compute_class_weight(class_weight='balanced', classes=pd.unique(y_train), y=y_train)
cw_dict = dict(zip(pd.unique(y_train), class_weights))

# Simpan scaler
joblib.dump(scaler, os.path.join(os.path.dirname(__file__), 'scaler_sleep.joblib'))

# Start MLflow run
with mlflow.start_run():
    clf = RandomForestClassifier(random_state=42, class_weight=cw_dict)
    clf.fit(X_train_scaled, y_train)
    joblib.dump(clf, os.path.join(os.path.dirname(__file__), 'model_sleep.joblib'))

print("[✓] Model selesai dilatih dan disimpan.")
