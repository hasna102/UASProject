import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from imblearn.over_sampling import SMOTE
import json

app = Flask(__name__)

# ─────────────────────────────────────────────
# Global variables (trained once on startup)
# ─────────────────────────────────────────────
models = {}
scaler = None
feature_columns = None
label_encoder = None
evaluation_results = {}


# ─────────────────────────────────────────────
# DATA GENERATION (synthetic, matches dataset)
# ─────────────────────────────────────────────
def generate_dataset(n=7500, seed=42):
    """
    Generate a synthetic dataset that mirrors the structure of
    'Smartphone Usage And Addiction Analysis 7500 Rows.csv'.
    """
    np.random.seed(seed)
    n = 7500

    age = np.random.randint(18, 36, n)
    gender = np.random.choice(['Male', 'Female', 'Other'], n)
    daily_screen_time = np.random.uniform(3, 12, n).round(1)
    social_media_hours = np.random.uniform(0.5, 6, n).round(1)
    gaming_hours = np.random.uniform(0, 5, n).round(1)
    work_study_hours = np.random.uniform(1, 8, n).round(1)
    sleep_hours = np.random.uniform(4.5, 9, n).round(1)
    notifications_per_day = np.random.randint(20, 251, n)
    app_opens_per_day = np.random.randint(15, 181, n)
    weekend_screen_time = (daily_screen_time * np.random.uniform(1.0, 1.5, n)).round(1)
    stress_level = np.random.choice(['Low', 'Medium', 'High'], n)
    academic_work_impact = np.random.choice(['Yes', 'No'], n)

    # Derive addiction_level based on screen time + social media
    score = (daily_screen_time * 0.4 + social_media_hours * 0.33 +
             weekend_screen_time * 0.27)
    addiction_level = np.where(score < 6.5, 'Mild',
                     np.where(score < 9.0, 'Moderate', 'Severe'))

    # Inject ~819 NaN in addiction_level (as per the report)
    nan_idx = np.random.choice(n, 819, replace=False)
    addiction_level = addiction_level.astype(object)
    addiction_level[nan_idx] = np.nan

    df = pd.DataFrame({
        'transaction_id': range(1, n + 1),
        'user_id': range(1001, n + 1001),
        'age': age,
        'gender': gender,
        'daily_screen_time_hours': daily_screen_time,
        'social_media_hours': social_media_hours,
        'gaming_hours': gaming_hours,
        'work_study_hours': work_study_hours,
        'sleep_hours': sleep_hours,
        'notifications_per_day': notifications_per_day,
        'app_opens_per_day': app_opens_per_day,
        'weekend_screen_time': weekend_screen_time,
        'stress_level': stress_level,
        'academic_work_impact': academic_work_impact,
        'addiction_level': addiction_level,
        'addicted_label': np.where(
            np.isin(addiction_level, ['Moderate', 'Severe']), 1, 0)
    })
    return df


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    # --- BAB 4: Cleaning Data ---
    df = df.dropna(subset=['addiction_level']).copy()

    # --- BAB 6: Encoding ---
    stress_map = {'Low': 0, 'Medium': 1, 'High': 2}
    impact_map = {'No': 0, 'Yes': 1}
    df['stress_level'] = df['stress_level'].map(stress_map)
    df['academic_work_impact'] = df['academic_work_impact'].map(impact_map)

    le = LabelEncoder()
    df['addiction_encoded'] = le.fit_transform(df['addiction_level'])  # Mild=0, Moderate=1, Severe=2

    feature_cols = [
        'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
        'work_study_hours', 'sleep_hours', 'notifications_per_day',
        'app_opens_per_day', 'weekend_screen_time',
        'stress_level', 'academic_work_impact', 'age'
    ]

    X = df[feature_cols].values
    y = df['addiction_encoded'].values

    # --- Normalisasi MinMax ---
    sc = MinMaxScaler()
    X_scaled = sc.fit_transform(X)

    # --- Train/Test Split 80:20 ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # --- SMOTE (balance train set) ---
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    return X_train_sm, X_test, y_train_sm, y_test, sc, feature_cols, le


# ─────────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────────
def train_all_models():
    global models, scaler, feature_columns, label_encoder, evaluation_results

    df = generate_dataset()
    X_train, X_test, y_train, y_test, sc, feat_cols, le = preprocess(df)

    scaler = sc
    feature_columns = feat_cols
    label_encoder = le

    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_dt = DecisionTreeClassifier(random_state=42)
    clf_knn = KNeighborsClassifier(n_neighbors=20)

    clf_rf.fit(X_train, y_train)
    clf_dt.fit(X_train, y_train)
    clf_knn.fit(X_train, y_train)

    models['Random Forest'] = clf_rf
    models['Decision Tree'] = clf_dt
    models['KNN'] = clf_knn

    # ── Evaluation ──
    results = {}
    class_names = ['Mild', 'Moderate', 'Severe']

    for name, clf in models.items():
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred,
                                       target_names=class_names,
                                       output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # 5-Fold CV
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')

        results[name] = {
            'accuracy': round(acc * 100, 2),
            'classification_report': report,
            'confusion_matrix': cm,
            'cv_mean': round(cv_scores.mean() * 100, 2),
            'cv_std': round(cv_scores.std() * 100, 2),
        }

    evaluation_results = results
    print("✅ All models trained successfully.")
    print({k: v['accuracy'] for k, v in results.items()})


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
                           results=evaluation_results,
                           feature_columns=feature_columns)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_features = [
            float(data['daily_screen_time_hours']),
            float(data['social_media_hours']),
            float(data['gaming_hours']),
            float(data['work_study_hours']),
            float(data['sleep_hours']),
            float(data['notifications_per_day']),
            float(data['app_opens_per_day']),
            float(data['weekend_screen_time']),
            int(data['stress_level']),        # 0/1/2
            int(data['academic_work_impact']),# 0/1
            int(data['age']),
        ]

        X_input = scaler.transform([input_features])
        class_names = ['Mild (Ringan)', 'Moderate (Sedang)', 'Severe (Parah)']
        predictions = {}

        for name, clf in models.items():
            pred = clf.predict(X_input)[0]
            proba = clf.predict_proba(X_input)[0].tolist()
            predictions[name] = {
                'prediction': class_names[pred],
                'probabilities': {
                    'Mild': round(proba[0] * 100, 1),
                    'Moderate': round(proba[1] * 100, 1),
                    'Severe': round(proba[2] * 100, 1),
                }
            }

        return jsonify({'success': True, 'predictions': predictions})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/evaluation')
def evaluation():
    return jsonify(evaluation_results)

# Train model saat aplikasi di-load
train_all_models()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)