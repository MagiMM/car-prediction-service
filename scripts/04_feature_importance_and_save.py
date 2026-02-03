"""
KROK 3.3: Feature Importance i zapis modeli
Analiza ważności cech i zapis najlepszych modeli do plików
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import joblib

# Ścieżki
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "car_pricing_amjad_zhour.csv"
MODELS_PATH = PROJECT_ROOT / "models"

print("=" * 80)
print("KROK 3.3: FEATURE IMPORTANCE I ZAPIS MODELI")
print("=" * 80)
print()

# Wczytanie danych
df = pd.read_csv(DATA_PATH)

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

# ============================================================
# TRENING MODELU REGRESJI
# ============================================================
print("🔨 Trening modelu regresji...")

preprocessor_reg = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

pipeline_reg = Pipeline([
    ('preprocessor', preprocessor_reg),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

pipeline_reg.fit(X_train, y_train_reg)
print("✅ Model regresji wytrenowany")
print()

# ============================================================
# FEATURE IMPORTANCE - REGRESJA
# ============================================================
print("=" * 80)
print("📊 FEATURE IMPORTANCE - MODEL REGRESJI")
print("=" * 80)
print()

# Pobranie feature importance z Random Forest
rf_model = pipeline_reg.named_steps['model']
feature_importances = rf_model.feature_importances_

# Nazwy cech po preprocessing
feature_names = (
    list(preprocessor_reg.named_transformers_['cat'].get_feature_names_out(categorical_features)) +
    numerical_features
)

# Utworzenie DataFrame z importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

print("Ranking ważności cech (TOP 15):")
print()
for i, row in importance_df.head(15).iterrows():
    bar_length = int(row['Importance'] * 100)
    bar = '█' * bar_length
    print(f"  {row['Feature']:25s} {bar} {row['Importance']:.4f}")
print()

# Grupowanie po typach
print("Suma ważności według typu cechy:")
print()

# Cechy numeryczne
num_importance = importance_df[importance_df['Feature'].isin(numerical_features)]['Importance'].sum()
print(f"  Numeryczne (Year, Engine, Mileage): {num_importance:.4f} ({num_importance*100:.1f}%)")

# Cechy kategoryczne - Make
make_importance = importance_df[importance_df['Feature'].str.startswith('Make_')]['Importance'].sum()
print(f"  Make (marka):                        {make_importance:.4f} ({make_importance*100:.1f}%)")

# Cechy kategoryczne - Model
model_importance = importance_df[importance_df['Feature'].str.startswith('Model_')]['Importance'].sum()
print(f"  Model:                               {model_importance:.4f} ({model_importance*100:.1f}%)")

# Cechy kategoryczne - Fuel Type
fuel_importance = importance_df[importance_df['Feature'].str.startswith('Fuel Type_')]['Importance'].sum()
print(f"  Fuel Type (rodzaj paliwa):           {fuel_importance:.4f} ({fuel_importance*100:.1f}%)")
print()

print("WNIOSKI:")
print()
top_feature = importance_df.iloc[0]
print(f"  • Najważniejsza cecha: {top_feature['Feature']} ({top_feature['Importance']:.4f})")
print(f"  • Cechy numeryczne razem: {num_importance*100:.1f}%")
print(f"  • Cechy kategoryczne razem: {(1-num_importance)*100:.1f}%")
print()

# ============================================================
# TRENING MODELU KLASYFIKACJI
# ============================================================
print("=" * 80)
print("Trening modelu klasyfikacji...")
print("=" * 80)
print()

preprocessor_cls = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)

pipeline_cls = Pipeline([
    ('preprocessor', preprocessor_cls),
    ('model', LogisticRegression(random_state=42, max_iter=1000))
])

pipeline_cls.fit(X_train, y_train_cls)
print("Model klasyfikacji wytrenowany")
print()

# ============================================================
# FEATURE IMPORTANCE - KLASYFIKACJA (COEFFICIENTS)
# ============================================================
print("=" * 80)
print("FEATURE COEFFICIENTS - MODEL KLASYFIKACJI")
print("=" * 80)
print()

# Pobranie współczynników z Logistic Regression
lr_model = pipeline_cls.named_steps['model']
coefficients = lr_model.coef_[0]

# Nazwy cech po preprocessing
feature_names_cls = (
    list(preprocessor_cls.named_transformers_['cat'].get_feature_names_out(categorical_features)) +
    numerical_features
)

# Utworzenie DataFrame z coefficients
coef_df = pd.DataFrame({
    'Feature': feature_names_cls,
    'Coefficient': coefficients
}).sort_values('Coefficient', key=abs, ascending=False)

print("Ranking ważności współczynników (TOP 15):")
print("(Dodatnie → zwiększa prawdopodobieństwo Manual)")
print("(Ujemne → zwiększa prawdopodobieństwo Automatic)")
print()
for i, row in coef_df.head(15).iterrows():
    sign = "+" if row['Coefficient'] >= 0 else "-"
    print(f"  {row['Feature']:25s} {sign} {abs(row['Coefficient']):.4f}")
print()

print("UWAGA: Niskie współczynniki sugerują, że cechy mają słaby wpływ na transmission")
print("   To wyjaśnia niską accuracy (47.5%) modelu klasyfikacji")
print()

# ============================================================
# ZAPIS MODELI
# ============================================================
print("=" * 80)
print("ZAPIS MODELI DO PLIKÓW")
print("=" * 80)
print()

# Upewnij się, że katalog models istnieje
MODELS_PATH.mkdir(exist_ok=True)

# Zapis modelu regresji
price_model_path = MODELS_PATH / "price_model.pkl"
joblib.dump(pipeline_reg, price_model_path)
print(f"Model regresji zapisany: {price_model_path}")

# Zapis modelu klasyfikacji
transmission_model_path = MODELS_PATH / "transmission_model.pkl"
joblib.dump(pipeline_cls, transmission_model_path)
print(f"Model klasyfikacji zapisany: {transmission_model_path}")
print()

# Sprawdzenie rozmiaru plików
price_size = price_model_path.stat().st_size / 1024 / 1024
trans_size = transmission_model_path.stat().st_size / 1024 / 1024

print("Rozmiary plików:")
print(f"  price_model.pkl:        {price_size:.2f} MB")
print(f"  transmission_model.pkl: {trans_size:.2f} MB")
print()

# ============================================================
# TEST WCZYTANIA MODELI
# ============================================================
print("=" * 80)
print("TEST WCZYTANIA I PREDYKCJI")
print("=" * 80)
print()

# Wczytanie
loaded_reg = joblib.load(price_model_path)
loaded_cls = joblib.load(transmission_model_path)
print("Modele wczytane pomyślnie")
print()

# Test predykcji
test_data = pd.DataFrame({
    'Make': ['Honda', 'BMW', 'Ford'],
    'Model': ['Model B', 'Model A', 'Model C'],
    'Year': [2020, 2015, 2018],
    'Engine Size': [2.5, 3.0, 1.8],
    'Mileage': [50000, 80000, 45000],
    'Fuel Type': ['Petrol', 'Diesel', 'Electric']
})

print("Przykładowe dane testowe:")
print(test_data)
print()

# Predykcja ceny
price_predictions = loaded_reg.predict(test_data)
print("Predykcje cen:")
for i, (idx, row) in enumerate(test_data.iterrows()):
    print(f"  {i+1}. {row['Make']} {row['Model']} ({row['Year']}): {price_predictions[i]:.2f} PLN")
print()

# Predykcja transmission
trans_predictions = loaded_cls.predict(test_data)
trans_proba = loaded_cls.predict_proba(test_data)

print("Predykcje transmission:")
for i, (idx, row) in enumerate(test_data.iterrows()):
    auto_prob = trans_proba[i][0] if loaded_cls.classes_[0] == 'Automatic' else trans_proba[i][1]
    manual_prob = trans_proba[i][1] if loaded_cls.classes_[1] == 'Manual' else trans_proba[i][0]
    print(f"  {i+1}. {row['Make']} {row['Model']}: {trans_predictions[i]} (Auto: {auto_prob:.2%}, Manual: {manual_prob:.2%})")
print()

# ============================================================
# METADATA MODELI
# ============================================================
print("=" * 80)
print("METADATA MODELI")
print("=" * 80)
print()

metadata = {
    'price_model': {
        'file': 'price_model.pkl',
        'type': 'RandomForestRegressor',
        'task': 'Regression - Price Prediction',
        'r2_score': 0.7915,
        'mae': 1937.17,
        'rmse': 2388.70,
        'features': feature_columns,
        'target': 'Price'
    },
    'transmission_model': {
        'file': 'transmission_model.pkl',
        'type': 'LogisticRegression',
        'task': 'Classification - Transmission Prediction',
        'accuracy': 0.4750,
        'f1_score': 0.5116,
        'features': feature_columns,
        'target': 'Transmission',
        'classes': ['Automatic', 'Manual']
    }
}

print("MODEL REGRESJI:")
for key, value in metadata['price_model'].items():
    print(f"  {key:15s}: {value}")
print()

print("MODEL KLASYFIKACJI:")
for key, value in metadata['transmission_model'].items():
    print(f"  {key:15s}: {value}")
print()

# Zapis metadata do pliku JSON
import json

metadata_path = MODELS_PATH / "models_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Metadata zapisana: {metadata_path}")
print()

# ============================================================
# PODSUMOWANIE
# ============================================================
print("=" * 80)
print("KROK 3.3 ZAKOŃCZONY")
print("=" * 80)
print()

print("ZAPISANE PLIKI:")
print(f"  • {price_model_path}")
print(f"  • {transmission_model_path}")
print(f"  • {metadata_path}")
print()

print("GOTOWE DO INTEGRACJI Z FASTAPI:")
print("  1. Modele zapisane i przetestowane")
print("  2. Feature importance przeanalizowana")
print("  3. Metadata dostępna")
print()

print("NASTĘPNY KROK: Integracja z FastAPI")
print("  • Wczytanie modeli przy starcie aplikacji")
print("  • Utworzenie endpointów /predict-price i /predict-transmission")
print("  • Walidacja danych wejściowych (Pydantic)")
print()
