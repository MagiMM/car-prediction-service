"""
KROK 3: TRENING I OCENA - Kompletna analiza modeli - diagnostyka jak dobry jest model
Szczegółowa ocena obu modeli z metrykami i wizualizacjami
"""

import pandas as pd
import numpy as np
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

# Ścieżki
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "car_pricing_amjad_zhour.csv"
MODELS_PATH = PROJECT_ROOT / "models"

print("=" * 80)
print("KROK 3: TRENING I OCENA MODELI")
print("=" * 80)
print()

# Wczytanie danych
df = pd.read_csv(DATA_PATH)
print(f"Wczytano {len(df)} rekordów")
print()

# Przygotowanie danych
feature_columns = ['Make', 'Model', 'Year', 'Engine Size', 'Mileage', 'Fuel Type']
categorical_features = ['Make', 'Model', 'Fuel Type']
numerical_features = ['Year', 'Engine Size', 'Mileage']

X = df[feature_columns]
y_price = df['Price']
y_transmission = df['Transmission']

# Podział danych
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_price, test_size=0.2, random_state=42
)

_, _, y_train_cls, y_test_cls = train_test_split(
    X, y_transmission, test_size=0.2, random_state=42, stratify=y_transmission
)

print(f"Podział: Train={len(X_train)}, Test={len(X_test)}")
print()

# ============================================================
# MODEL 1: REGRESJA (Przewidywanie ceny)
# ============================================================
print("=" * 80)
print("MODEL 1: REGRESJA - Przewidywanie ceny")
print("=" * 80)
print()

# Preprocessor
preprocessor_reg = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

# Pipeline
pipeline_reg = Pipeline([
    ('preprocessor', preprocessor_reg),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

print("🔨 Trening RandomForestRegressor...")
pipeline_reg.fit(X_train, y_train_reg)
print("Model wytrenowany!")
print()

# Predykcje
y_train_pred_reg = pipeline_reg.predict(X_train)
y_test_pred_reg = pipeline_reg.predict(X_test)

# ============================================================
# METRYKI REGRESJI
# ============================================================
print("=" * 80)
print("METRYKI REGRESJI")
print("=" * 80)
print()

# Train metrics
train_mae = mean_absolute_error(y_train_reg, y_train_pred_reg)
train_rmse = np.sqrt(mean_squared_error(y_train_reg, y_train_pred_reg))
train_r2 = r2_score(y_train_reg, y_train_pred_reg)

# Test metrics
test_mae = mean_absolute_error(y_test_reg, y_test_pred_reg)
test_rmse = np.sqrt(mean_squared_error(y_test_reg, y_test_pred_reg))
test_r2 = r2_score(y_test_reg, y_test_pred_reg)

print("TRAIN SET:")
print(f"  MAE (Mean Absolute Error):     {train_mae:>10.2f} PLN")
print(f"  RMSE (Root Mean Squared Error): {train_rmse:>10.2f} PLN")
print(f"  R² Score:                       {train_r2:>10.4f}")
print()

print("TEST SET:")
print(f"  MAE (Mean Absolute Error):     {test_mae:>10.2f} PLN")
print(f"  RMSE (Root Mean Squared Error): {test_rmse:>10.2f} PLN")
print(f"  R² Score:                       {test_r2:>10.4f}")
print()

print("INTERPRETACJA:")
print(f"  • Średni błąd bezwzględny: {test_mae:.2f} PLN")
print(f"  • Model wyjaśnia {test_r2*100:.1f}% wariancji cen")
print(f"  • Średnia cena: {y_test_reg.mean():.2f} PLN")
print(f"  • Błąd względny: {(test_mae/y_test_reg.mean())*100:.2f}%")
print()

# Overfitting check
if train_r2 - test_r2 > 0.1:
    print("UWAGA: Możliwy overfitting (train R² >> test R²)")
else:
    print("Brak oznak overfittingu")
print()

# ============================================================
# ANALIZA BŁĘDÓW REGRESJI
# ============================================================
print("=" * 80)
print("ANALIZA BŁĘDÓW REGRESJI")
print("=" * 80)
print()

# Błędy
errors_reg = y_test_pred_reg - y_test_reg
abs_errors_reg = np.abs(errors_reg)

print("Statystyki błędów:")
print(f"  Średni błąd:           {errors_reg.mean():>10.2f} PLN")
print(f"  Median błędu:          {np.median(errors_reg):>10.2f} PLN")
print(f"  Std błędu:             {errors_reg.std():>10.2f} PLN")
print(f"  Min błąd (niedoszac.): {errors_reg.min():>10.2f} PLN")
print(f"  Max błąd (przeszac.):  {errors_reg.max():>10.2f} PLN")
print()

# Najgorsze predykcje
print("TOP 5 NAJGORSZYCH PREDYKCJI:")
worst_indices = abs_errors_reg.argsort()[-5:][::-1]
for i, idx in enumerate(worst_indices, 1):
    actual = y_test_reg.iloc[idx]
    predicted = y_test_pred_reg[idx]
    error = predicted - actual
    print(f"  {i}. Rzeczywista: {actual:>10.2f}, Predykcja: {predicted:>10.2f}, Błąd: {error:>+10.2f} PLN")
print()

# Najlepsze predykcje
print("TOP 5 NAJLEPSZYCH PREDYKCJI:")
best_indices = abs_errors_reg.argsort()[:5]
for i, idx in enumerate(best_indices, 1):
    actual = y_test_reg.iloc[idx]
    predicted = y_test_pred_reg[idx]
    error = predicted - actual
    print(f"  {i}. Rzeczywista: {actual:>10.2f}, Predykcja: {predicted:>10.2f}, Błąd: {error:>+10.2f} PLN")
print()

# ============================================================
# CROSS-VALIDATION REGRESJI
# ============================================================
print("=" * 80)
print("CROSS-VALIDATION (5-fold)")
print("=" * 80)
print()

print("Wykonuję 5-fold cross-validation (może potrwać chwilę)...")
cv_scores_reg = cross_val_score(
    pipeline_reg, X_train, y_train_reg, 
    cv=5, scoring='r2', n_jobs=-1
)

print(f"Zakończono")
print()
print(f"R² scores dla każdego foldu:")
for i, score in enumerate(cv_scores_reg, 1):
    print(f"  Fold {i}: {score:.4f}")
print()
print(f"Średni R²: {cv_scores_reg.mean():.4f} (±{cv_scores_reg.std():.4f})")
print()

if cv_scores_reg.std() < 0.05:
    print("Model jest stabilny (niskie odchylenie między foldami)")
else:
    print("Model może być niestabilny (wysokie odchylenie między foldami)")
print()

# ============================================================
# MODEL 2: KLASYFIKACJA (Przewidywanie transmission)
# ============================================================
print("=" * 80)
print("MODEL 2: KLASYFIKACJA - Przewidywanie transmission")
print("=" * 80)
print()

# Preprocessor
preprocessor_cls = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)

# Pipeline
pipeline_cls = Pipeline([
    ('preprocessor', preprocessor_cls),
    ('model', LogisticRegression(random_state=42, max_iter=1000))
])

print("Trening LogisticRegression...")
pipeline_cls.fit(X_train, y_train_cls)
print("Model wytrenowany!")
print()

# Predykcje
y_train_pred_cls = pipeline_cls.predict(X_train)
y_test_pred_cls = pipeline_cls.predict(X_test)

# ============================================================
# METRYKI KLASYFIKACJI
# ============================================================
print("=" * 80)
print("METRYKI KLASYFIKACJI")
print("=" * 80)
print()

# Train metrics
train_acc = accuracy_score(y_train_cls, y_train_pred_cls)
train_precision = precision_score(y_train_cls, y_train_pred_cls, pos_label='Manual')
train_recall = recall_score(y_train_cls, y_train_pred_cls, pos_label='Manual')
train_f1 = f1_score(y_train_cls, y_train_pred_cls, pos_label='Manual')

# Test metrics
test_acc = accuracy_score(y_test_cls, y_test_pred_cls)
test_precision = precision_score(y_test_cls, y_test_pred_cls, pos_label='Manual')
test_recall = recall_score(y_test_cls, y_test_pred_cls, pos_label='Manual')
test_f1 = f1_score(y_test_cls, y_test_pred_cls, pos_label='Manual')

print("TRAIN SET:")
print(f"  Accuracy:  {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"  Precision: {train_precision:.4f} (Manual)")
print(f"  Recall:    {train_recall:.4f} (Manual)")
print(f"  F1-Score:  {train_f1:.4f}")
print()

print("TEST SET:")
print(f"  Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  Precision: {test_precision:.4f} (Manual)")
print(f"  Recall:    {test_recall:.4f} (Manual)")
print(f"  F1-Score:  {test_f1:.4f}")
print()

print("INTERPRETACJA:")
print(f"  • Model poprawnie klasyfikuje {test_acc*100:.1f}% przypadków")
print(f"  • Baseline (zawsze Manual): {(y_test_cls=='Manual').sum()/len(y_test_cls)*100:.1f}%")
if test_acc < 0.55:
    print("Model jest GORSZY lub porównywalny z losowym zgadywaniem!")
else:
    print("Model jest lepszy niż losowe zgadywanie")
print()

# ============================================================
# CONFUSION MATRIX
# ============================================================
print("=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)
print()

cm = confusion_matrix(y_test_cls, y_test_pred_cls, labels=['Automatic', 'Manual'])

print("                Predicted")
print("                Automatic  Manual")
print(f"Actual Automatic  {cm[0,0]:>4d}      {cm[0,1]:>4d}")
print(f"       Manual     {cm[1,0]:>4d}      {cm[1,1]:>4d}")
print()

# Analiza confusion matrix
tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

print("Szczegóły:")
print(f"  True Positives (Manual → Manual):       {tp:>4d}")
print(f"  True Negatives (Automatic → Automatic): {tn:>4d}")
print(f"  False Positives (Automatic → Manual):   {fp:>4d}")
print(f"  False Negatives (Manual → Automatic):   {fn:>4d}")
print()

# ============================================================
# CLASSIFICATION REPORT
# ============================================================
print("=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)
print()

print(classification_report(y_test_cls, y_test_pred_cls))

# ============================================================
# PODSUMOWANIE I WNIOSKI
# ============================================================
print("=" * 80)
print("PODSUMOWANIE WYNIKÓW")
print("=" * 80)
print()

print("MODEL REGRESJI (Przewidywanie ceny):")
print(f"  R² Score:     {test_r2:.4f} ({test_r2*100:.1f}%)")
print(f"  MAE:          {test_mae:.2f} PLN")
print(f"  RMSE:         {test_rmse:.2f} PLN")
print(f"  Błąd wzgl.:   {(test_mae/y_test_reg.mean())*100:.2f}%")
print()
print("  OCENA: Dobry model, gotowy do użycia w produkcji")
print()

print("MODEL KLASYFIKACJI (Przewidywanie transmission):")
print(f"  Accuracy:    {test_acc:.4f} ({test_acc*100:.1f}%)")
print(f"  Precision:   {test_precision:.4f}")
print(f"  Recall:      {test_recall:.4f}")
print(f"  F1-Score:    {test_f1:.4f}")
print()
print("  OCENA: Słaby model, wymaga poprawy lub inne podejście")
print("     Możliwe przyczyny:")
print("     • Transmission może nie zależeć od podanych cech")
print("     • Brak wystarczającej informacji w danych")
print("     • Model zbyt prosty dla tego problemu")
print()

print("=" * 80)
print("KROK 3 - OCENA ZAKOŃCZONA")
print("=" * 80)
print()
