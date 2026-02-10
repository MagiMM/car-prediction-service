"""
Model Training & Evaluation Pipeline
- Train two ML models: Price Prediction (Regression) & Transmission Classification
- Evaluate performance with comprehensive metrics
- Analyze feature importance
- Save trained models for production use
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

# ============================================================
# SETUP
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "car_pricing_amjad_zhour.csv"
MODELS_PATH = PROJECT_ROOT / "models"
MODELS_PATH.mkdir(exist_ok=True)

print("=" * 80)
print("MODEL TRAINING PIPELINE")
print("=" * 80)
print()

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} records")
print()

# ============================================================
# FEATURE ENGINEERING
# ============================================================
feature_columns = ['Make', 'Model', 'Year', 'Engine Size', 'Mileage', 'Fuel Type']
categorical_features = ['Make', 'Model', 'Fuel Type']
numerical_features = ['Year', 'Engine Size', 'Mileage']

X = df[feature_columns]
y_price = df['Price']  # REGRESJA: Target
y_transmission = df['Transmission']  # KLASYFIKACJA: Target

# ============================================================
# TRAIN/TEST SPLIT
# ============================================================
# REGRESJA: Split for price prediction
X_train, X_test, y_train_price, y_test_price = train_test_split(
    X, y_price, test_size=0.2, random_state=42
)

# KLASYFIKACJA: Split with stratification
_, _, y_train_transmission, y_test_transmission = train_test_split(
    X, y_transmission, test_size=0.2, random_state=42, stratify=y_transmission
)

print(f"Train/Test Split: {len(X_train)} / {len(X_test)}")
print()


# ============================================================
# REGRESJA: MODEL 1 - PRICE PREDICTION
# ============================================================
print("=" * 80)
print("MODEL 1: PRICE PREDICTION (REGRESSION)")
print("=" * 80)
print()

# REGRESJA: Preprocessor - OneHot for categorical, passthrough for numerical (RF doesn't need scaling)
preprocessor_reg = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

# REGRESJA: Pipeline with RandomForestRegressor
pipeline_reg = Pipeline([
    ('preprocessor', preprocessor_reg),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# REGRESJA: Train
print("Training RandomForestRegressor...")
pipeline_reg.fit(X_train, y_train_price)
print("Model trained")
print()

# REGRESJA: Predict
y_train_pred = pipeline_reg.predict(X_train)
y_test_pred = pipeline_reg.predict(X_test)

# REGRESJA: Evaluate
train_r2 = r2_score(y_train_price, y_train_pred)
test_mae = mean_absolute_error(y_test_price, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test_price, y_test_pred))
test_r2 = r2_score(y_test_price, y_test_pred)

print("REGRESJA METRICS:")
print(f"  Train R²:  {train_r2:.4f}")
print(f"  Test R²:   {test_r2:.4f}")
print(f"  Test MAE:  {test_mae:.2f} PLN")
print(f"  Test RMSE: {test_rmse:.2f} PLN")
print(f"  Relative Error: {(test_mae/y_test_price.mean())*100:.2f}%")
print()

# Overfitting check
if train_r2 - test_r2 > 0.1:
    print("Warning: Possible overfitting detected")
else:
    print("No overfitting detected")
print()

# REGRESJA: Error analysis
errors = y_test_pred - y_test_price
abs_errors = np.abs(errors)

print("ERROR ANALYSIS:")
print(f"  Mean Error:   {errors.mean():>8.2f} PLN")
print(f"  Median Error: {np.median(errors):>8.2f} PLN")
print(f"  Std Dev:      {errors.std():>8.2f} PLN")
print(f"  Min Error:    {errors.min():>8.2f} PLN")
print(f"  Max Error:    {errors.max():>8.2f} PLN")
print()

# Worst predictions
print("TOP 3 WORST PREDICTIONS:")
worst_idx = abs_errors.argsort()[-3:][::-1]
for i, idx in enumerate(worst_idx, 1):
    actual = y_test_price.iloc[idx]
    predicted = y_test_pred[idx]
    error = predicted - actual
    print(f"  {i}. Actual: {actual:>8.2f}, Predicted: {predicted:>8.2f}, Error: {error:>+8.2f} PLN")
print()

# REGRESJA: Cross-validation
print("Cross-Validation (5-fold)...")
cv_scores = cross_val_score(pipeline_reg, X_train, y_train_price, cv=5, scoring='r2', n_jobs=-1)
print(f"  CV R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print()

# REGRESJA: Feature Importance
print("FEATURE IMPORTANCE (TOP 10):")
rf_model = pipeline_reg.named_steps['model']
feature_importances = rf_model.feature_importances_
feature_names = (
    list(preprocessor_reg.named_transformers_['cat'].get_feature_names_out(categorical_features)) +
    numerical_features
)
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

for i, row in importance_df.head(10).iterrows():
    bar = '█' * int(row['Importance'] * 100)
    print(f"  {row['Feature']:30s} {bar} {row['Importance']:.4f}")
print()


# ============================================================
# KLASYFIKACJA: MODEL 2 - TRANSMISSION PREDICTION
# ============================================================
print("=" * 80)
print("MODEL 2: TRANSMISSION PREDICTION (CLASSIFICATION)")
print("=" * 80)
print()

# KLASYFIKACJA: Preprocessor with StandardScaler (LogReg needs scaling)
preprocessor_cls = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)

# KLASYFIKACJA: Pipeline with LogisticRegression
pipeline_cls = Pipeline([
    ('preprocessor', preprocessor_cls),
    ('model', LogisticRegression(random_state=42, max_iter=1000))
])

# KLASYFIKACJA: Train
print("Training LogisticRegression...")
pipeline_cls.fit(X_train, y_train_transmission)
print("Model trained")
print()

# KLASYFIKACJA: Predict
y_train_pred_cls = pipeline_cls.predict(X_train)
y_test_pred_cls = pipeline_cls.predict(X_test)

# KLASYFIKACJA: Evaluate
train_acc = accuracy_score(y_train_transmission, y_train_pred_cls)
test_acc = accuracy_score(y_test_transmission, y_test_pred_cls)
test_precision = precision_score(y_test_transmission, y_test_pred_cls, pos_label='Manual')
test_recall = recall_score(y_test_transmission, y_test_pred_cls, pos_label='Manual')
test_f1 = f1_score(y_test_transmission, y_test_pred_cls, pos_label='Manual')

print("CLASSIFICATION METRICS:")
print(f"  Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"  Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  Precision:      {test_precision:.4f}")
print(f"  Recall:         {test_recall:.4f}")
print(f"  F1-Score:       {test_f1:.4f}")
print()

# KLASYFIKACJA: Baseline comparison
baseline_acc = (y_test_transmission == 'Manual').sum() / len(y_test_transmission)
print(f"  Baseline (always predict Manual): {baseline_acc*100:.1f}%")
if test_acc > baseline_acc:
    print(f"  Model is better than baseline by {(test_acc - baseline_acc)*100:.1f}%")
else:
    print("  Model is not better than baseline!")
print()

# KLASYFIKACJA: Confusion Matrix
cm = confusion_matrix(y_test_transmission, y_test_pred_cls, labels=['Automatic', 'Manual'])

print("CONFUSION MATRIX:")
print("                Predicted")
print("                Automatic  Manual")
print(f"Actual Automatic  {cm[0,0]:>4d}      {cm[0,1]:>4d}")
print(f"       Manual     {cm[1,0]:>4d}      {cm[1,1]:>4d}")
print()

tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
print(f"  True Positives:  {tp:>4d}  (Manual -> Manual)")
print(f"  True Negatives:  {tn:>4d}  (Automatic -> Automatic)")
print(f"  False Positives: {fp:>4d}  (Automatic -> Manual)")
print(f"  False Negatives: {fn:>4d}  (Manual -> Automatic)")
print()

# KLASYFIKACJA: Classification Report
print("CLASSIFICATION REPORT:")
print(classification_report(y_test_transmission, y_test_pred_cls))

# KLASYFIKACJA: Feature Coefficients
print("FEATURE COEFFICIENTS (TOP 10):")
lr_model = pipeline_cls.named_steps['model']
coefficients = lr_model.coef_[0]
feature_names_cls = (
    list(preprocessor_cls.named_transformers_['cat'].get_feature_names_out(categorical_features)) +
    numerical_features
)
coef_df = pd.DataFrame({
    'Feature': feature_names_cls,
    'Coefficient': coefficients
}).sort_values('Coefficient', key=abs, ascending=False)

print("(Positive -> increases Manual probability, Negative -> increases Automatic probability)")
for i, row in coef_df.head(10).iterrows():
    sign = "+" if row['Coefficient'] >= 0 else ""
    print(f"  {row['Feature']:30s} {sign}{row['Coefficient']:>8.4f}")
print()


# ============================================================
# SAVE MODELS
# ============================================================
print("=" * 80)
print("SAVING MODELS")
print("=" * 80)
print()

# REGRESJA: Save regression model
price_model_path = MODELS_PATH / "price_model.pkl"
joblib.dump(pipeline_reg, price_model_path)
price_size = price_model_path.stat().st_size / 1024 / 1024
print(f"Regression model saved: {price_model_path} ({price_size:.2f} MB)")

# KLASYFIKACJA: Save classification model
transmission_model_path = MODELS_PATH / "transmission_model.pkl"
joblib.dump(pipeline_cls, transmission_model_path)
trans_size = transmission_model_path.stat().st_size / 1024 / 1024
print(f"Classification model saved: {transmission_model_path} ({trans_size:.2f} MB)")
print()


# ============================================================
# TEST LOADING & PREDICTIONS
# ============================================================
print("=" * 80)
print("TESTING MODEL LOADING")
print("=" * 80)
print()

# Load models
loaded_reg = joblib.load(price_model_path)  # REGRESJA
loaded_cls = joblib.load(transmission_model_path)  # KLASYFIKACJA
print("Models loaded successfully")
print()

# Test predictions
test_data = pd.DataFrame({
    'Make': ['Honda', 'BMW', 'Ford'],
    'Model': ['Model B', 'Model A', 'Model C'],
    'Year': [2020, 2015, 2018],
    'Engine Size': [2.5, 3.0, 1.8],
    'Mileage': [50000, 80000, 45000],
    'Fuel Type': ['Petrol', 'Diesel', 'Electric']
})

print("Test Data:")
print(test_data.to_string(index=False))
print()

# REGRESJA: Price predictions
price_preds = loaded_reg.predict(test_data)
print("Price Predictions:")
for i, (_, row) in enumerate(test_data.iterrows()):
    print(f"  {row['Make']} {row['Model']} ({row['Year']}): {price_preds[i]:,.2f} PLN")
print()

# KLASYFIKACJA: Transmission predictions
trans_preds = loaded_cls.predict(test_data)
trans_proba = loaded_cls.predict_proba(test_data)

print("Transmission Predictions:")
for i, (_, row) in enumerate(test_data.iterrows()):
    auto_idx = 0 if loaded_cls.classes_[0] == 'Automatic' else 1
    manual_idx = 1 - auto_idx
    auto_prob = trans_proba[i][auto_idx]
    manual_prob = trans_proba[i][manual_idx]
    print(f"  {row['Make']} {row['Model']}: {trans_preds[i]} (Auto: {auto_prob:.1%}, Manual: {manual_prob:.1%})")
print()


# ============================================================
# SAVE METADATA
# ============================================================
metadata = {
    'price_model': {  # REGRESJA
        'file': 'price_model.pkl',
        'type': 'RandomForestRegressor',
        'task': 'Regression',
        'target': 'Price',
        'metrics': {
            'r2_score': float(test_r2),
            'mae': float(test_mae),
            'rmse': float(test_rmse),
            'relative_error': float((test_mae/y_test_price.mean())*100)
        },
        'features': feature_columns,
        'n_features': len(feature_columns)
    },
    'transmission_model': {  # KLASYFIKACJA
        'file': 'transmission_model.pkl',
        'type': 'LogisticRegression',
        'task': 'Classification',
        'target': 'Transmission',
        'classes': list(loaded_cls.classes_),
        'metrics': {
            'accuracy': float(test_acc),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'f1_score': float(test_f1)
        },
        'features': feature_columns,
        'n_features': len(feature_columns)
    }
}

metadata_path = MODELS_PATH / "models_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Metadata saved: {metadata_path}")
print()


# ============================================================
# SUMMARY
# ============================================================
print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print()

print("REGRESJA - REGRESSION MODEL:")
print(f"  R² Score:  {test_r2:.4f} ({test_r2*100:.1f}%)")
print(f"  MAE:       {test_mae:.2f} PLN")
print(f"  Status:    Ready for production")
print()

print("KLASYFIKACJA - CLASSIFICATION MODEL:")
print(f"  Accuracy:  {test_acc:.4f} ({test_acc*100:.1f}%)")
print(f"  F1-Score:  {test_f1:.4f}")
print(f"  Status:    {'Needs improvement' if test_acc < 0.6 else 'Ready for production'}")
print()

print("SAVED FILES:")
print(f"  - {price_model_path.name}")
print(f"  - {transmission_model_path.name}")
print(f"  - {metadata_path.name}")
print()

print("NEXT STEPS:")
print("  - Integrate models with FastAPI endpoints")
print("  - Add input validation with Pydantic schemas")
print("  - Deploy to production")
print()
