"""
KROK 2.5: Pipeline - kompletny pipeline preprocessing + model. 
Przykładowo [Dane surowe] → [Encoding] + [Scaling] → [Model] → [Predykcja]
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

# Ścieżki
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "car_pricing_amjad_zhour.csv"

print("=" * 80)
print("KROK 2.5: PIPELINE")
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

print("=" * 80)
print("STRUKTURA DANYCH")
print("=" * 80)
print()
print(f"Cechy kategoryczne (tekst): {categorical_features}")
print(f"Cechy numeryczne:           {numerical_features}")
print(f"Target regresji:             Price")
print(f"Target klasyfikacji:         Transmission")
print()

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
# COLUMN TRANSFORMER - Stosuje różne transformacje do różnych kolumn. 
# Kategoryczne potrzebują OneHot, numeryczne czasem skalowania, czasem nie.
# ============================================================
print("=" * 80)
print("COLUMN TRANSFORMER")
print("=" * 80)
print()

print("Różne kolumny wymagają różnych transformacji!")
print("  • Kategoryczne (Make, Model) → OneHotEncoder")
print("  • Numeryczne (Year, Mileage) → StandardScaler (tylko dla niektórych modeli)")
print()

print("ColumnTransformer pozwala aplikować różne transformery do różnych kolumn:")
print()

# Preprocessor dla REGRESJI (bez scalera dla RandomForest)
preprocessor_regression = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
        ('num', 'passthrough', numerical_features)  # 'passthrough' = pozostaw bez zmian
    ]
)

print("PREPROCESSOR DLA REGRESJI:")
print("  • Kategoryczne → OneHotEncoder(drop='first')")
print("  • Numeryczne → passthrough (bez zmian)")
print()

# Preprocessor dla KLASYFIKACJI (ze scalerem dla LogisticRegression)
preprocessor_classification = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
        ('num', StandardScaler(), numerical_features)  # Skalowanie dla Logistic
    ]
)

print("PREPROCESSOR DLA KLASYFIKACJI:")
print("  • Kategoryczne → OneHotEncoder(drop='first')")
print("  • Numeryczne → StandardScaler()")
print()

# ============================================================
# PIPELINE 1: REGRESJA
# ============================================================
print("=" * 80)
print("PIPELINE 1: MODEL REGRESJI (Przewidywanie ceny)")
print("=" * 80)
print()

# Utworzenie pipeline
pipeline_regression = Pipeline([
    ('preprocessor', preprocessor_regression),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

print("Struktura Pipeline:")
print("  1. preprocessor (ColumnTransformer)")
print("     ├─ cat: OneHotEncoder")
print("     └─ num: passthrough")
print("  2. model (RandomForestRegressor)")
print()

print("Trening modelu...")
pipeline_regression.fit(X_train, y_train_reg)
print("Model wytrenowany!")
print()

# Predykcja
train_score_reg = pipeline_regression.score(X_train, y_train_reg)
test_score_reg = pipeline_regression.score(X_test, y_test_reg)

print(f"Wyniki (R² score):")
print(f"  Train: {train_score_reg:.4f}")
print(f"  Test:  {test_score_reg:.4f}")
print()

# Przykładowa predykcja
sample_X = X_test.iloc[:3]
predictions_reg = pipeline_regression.predict(sample_X)
actual_reg = y_test_reg.iloc[:3].values

print("Przykładowe predykcje (pierwsze 3 z test set):")
for i in range(3):
    print(f"  {i+1}. Predykcja: {predictions_reg[i]:.2f}, Rzeczywista: {actual_reg[i]:.2f}, Różnica: {abs(predictions_reg[i] - actual_reg[i]):.2f}")
print()

# ============================================================
# PIPELINE 2: KLASYFIKACJA
# ============================================================
print("=" * 80)
print("PIPELINE 2: MODEL KLASYFIKACJI (Przewidywanie transmission)")
print("=" * 80)
print()

# Utworzenie pipeline
pipeline_classification = Pipeline([
    ('preprocessor', preprocessor_classification),
    ('model', LogisticRegression(random_state=42, max_iter=1000))
])

print("Struktura Pipeline:")
print("  1. preprocessor (ColumnTransformer)")
print("     ├─ cat: OneHotEncoder")
print("     └─ num: StandardScaler")
print("  2. model (LogisticRegression)")
print()

print("Trening modelu...")
pipeline_classification.fit(X_train, y_train_cls)
print("Model wytrenowany!")
print()

# Predykcja
train_score_cls = pipeline_classification.score(X_train, y_train_cls)
test_score_cls = pipeline_classification.score(X_test, y_test_cls)

print(f"Wyniki (Accuracy):")
print(f"  Train: {train_score_cls:.4f} ({train_score_cls*100:.2f}%)")
print(f"  Test:  {test_score_cls:.4f} ({test_score_cls*100:.2f}%)")
print()

# Przykładowa predykcja
sample_X_cls = X_test.iloc[:5]
predictions_cls = pipeline_classification.predict(sample_X_cls)
actual_cls = y_test_cls.iloc[:5].values

print("Przykładowe predykcje (pierwsze 5 z test set):")
for i in range(5):
    correct = "ok" if predictions_cls[i] == actual_cls[i] else "źle"
    print(f"  {i+1}. Predykcja: {predictions_cls[i]:10s}, Rzeczywista: {actual_cls[i]:10s} {correct}")
print()


# ============================================================
# JAK UŻYWAĆ PIPELINE W PRODUKCJI?
# ============================================================
print("=" * 80)
print(" JAK UŻYWAĆ PIPELINE W PRODUKCJI?")
print("=" * 80)
print()

print("1. ZAPIS DO PLIKU:")
print("   import joblib")
print("   joblib.dump(pipeline_regression, 'models/price_model.pkl')")
print()

print("2. WCZYTANIE W FASTAPI:")
print("   model = joblib.load('models/price_model.pkl')")
print()

print("3. PREDYKCJA:")
print("   new_data = pd.DataFrame({")
print("       'Make': ['Honda'], 'Model': ['Model B'],")
print("       'Year': [2020], 'Engine Size': [2.5],")
print("       'Mileage': [50000], 'Fuel Type': ['Petrol']")
print("   })")
print("   price = model.predict(new_data)")
print()
print("   Pipeline automatycznie:")
print("      • Zakoduje kategorie (OneHot)")
print("      • Przeskaluje (jeśli potrzeba)")
print("      • Zwróci predykcję")
print()

# ============================================================
# PODSUMOWANIE
# ============================================================
print("=" * 80)
print(" KROK 2.5 ZAKOŃCZONY - PREPROCESSING KOMPLETNY!")
print("=" * 80)
print()

print(" CO OSIĄGNĘLIŚMY:")
print()
print("PIPELINE REGRESJI:")
print("  OneHotEncoder dla kategorii")
print("  Bez skalowania (RandomForest nie wymaga)")
print("  RandomForestRegressor(n_estimators=100)")
print(f"  R² score test: {test_score_reg:.4f}")
print()

print("PIPELINE KLASYFIKACJI:")
print("  OneHotEncoder dla kategorii")
print("  StandardScaler dla liczb")
print("  LogisticRegression")
print(f"  Accuracy test: {test_score_cls:.4f} ({test_score_cls*100:.2f}%)")
print()

print(" TERAZ GOTOWE DO:")
print("  1. Trening pełnych modeli z optymalizacją")
print("  2. Szczegółowa ocena (metryki, confusion matrix, etc.)")
print("  3. Zapis modeli do plików")
print("  4. Integracja z FastAPI")
print()

print("=" * 80)
print(" PREPROCESSING ZAKOŃCZONY - GOTOWI DO KROKU 3: TRENING I OCENA")
print("=" * 80)
print()
