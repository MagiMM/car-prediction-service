"""
KROK 2.3: Skalowanie (StandardScaler)
Pokazuje jak standaryzować dane i kiedy jest to potrzebne
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Ścieżki
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "car_pricing_amjad_zhour.csv"

print("=" * 80)
print("KROK 2.3: SKALOWANIE (StandardScaler) - STANDARYZACJA DANYCH NUMERYCZNYCH")
print("=" * 80)
print()

# Wczytanie danych
df = pd.read_csv(DATA_PATH)
print(f"Wczytano {len(df)} rekordów")
print()

# Wybieramy kolumny numeryczne do skalowania
numerical_columns = ['Year', 'Engine Size', 'Mileage']

print("=" * 80)
print("DANE PRZED SKALOWANIEM")
print("=" * 80)
print()
print("Przykładowe 20 wierszy:")
print(df[numerical_columns].head(20))
print()

# Statystyki PRZED
print("Statystyki PRZED skalowaniem:")
print()
for col in numerical_columns:
    print(f"{col:15s}:")
    print(f"  Zakres:  {df[col].min():>10.1f} - {df[col].max():>10.1f}")
    print(f"  Średnia: {df[col].mean():>10.2f}")
    print(f"  Std dev: {df[col].std():>10.2f}")
    print()

print("PROBLEM:")
print("  • Mileage ma zakres ~200,000 (min: 56, max: 199,867)")
print("  • Year ma zakres ~21 (min: 2000, max: 2021)")
print("  • Engine Size ma zakres ~3.5 (min: 1.0, max: 4.5)")
print()
print("  Mileage jest 10,000x większy niż Engine Size!")
print()

# ============================================================
# ZASTOSOWANIE StandardScaler
# ============================================================
print("=" * 80)
print("ZASTOSOWANIE StandardScaler")
print("=" * 80)
print()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numerical_columns])

# Utworzenie DataFrame ze skalowanymi danymi
df_scaled = pd.DataFrame(
    scaled_data, 
    columns=[f"{col}_scaled" for col in numerical_columns]
)

print("CO ROBI StandardScaler?")
print()
print("  Formula: X_scaled = (X - μ) / σ")
print()
print("  gdzie:")
print("    X       = oryginalna wartość")
print("    μ (mu)  = średnia kolumny")
print("    σ (sigma) = odchylenie standardowe")
print()
print("  Rezultat: Wszystkie kolumny mają średnią=0 i std=1")
print()

# ============================================================
# DANE PO SKALOWANIU
# ============================================================
print("=" * 80)
print("DANE PO SKALOWANIU")
print("=" * 80)
print()
print("Przykładowe 10 wierszy:")
print(df_scaled.head(10))
print()

# Statystyki PO
print("Statystyki PO skalowaniu:")
print()
for i, col in enumerate(numerical_columns):
    scaled_col = df_scaled.columns[i]
    print(f"{col:15s} → {scaled_col}:")
    print(f"  Zakres:  {df_scaled[scaled_col].min():>10.4f} - {df_scaled[scaled_col].max():>10.4f}")
    print(f"  Średnia: {df_scaled[scaled_col].mean():>10.6f}  ← ~0")
    print(f"  Std dev: {df_scaled[scaled_col].std():>10.6f}  ← ~1")
    print()

print("Wszystkie kolumny są teraz w podobnej skali!")
print()

# ============================================================
# PORÓWNANIE PRZED/PO
# ============================================================
print("=" * 80)
print("PORÓWNANIE: Jeden rekord PRZED vs PO")
print("=" * 80)
print()

# Wybierzmy pierwszy rekord
idx = 0
print(f"Rekord #{idx + 1}:")
print()
print("PRZED skalowaniem:")
for col in numerical_columns:
    print(f"  {col:15s}: {df[col].iloc[idx]:>10.2f}")
print()

print("PO skalowaniu:")
for i, col in enumerate(numerical_columns):
    scaled_col = df_scaled.columns[i]
    original = df[col].iloc[idx]
    scaled = df_scaled[scaled_col].iloc[idx]
    mean = df[col].mean()
    std = df[col].std()
    
    print(f"  {col:15s}: {scaled:>10.4f}")
    print(f"    └─ Obliczenie: ({original:.2f} - {mean:.2f}) / {std:.2f} = {scaled:.4f}")
print()

# ============================================================
# INTERPRETACJA WARTOŚCI SKALOWANYCH
# ============================================================
print("=" * 80)
print("JAK INTERPRETOWAĆ WARTOŚCI SKALOWANE?")
print("=" * 80)
print()
print("Wartość skalowana to 'liczba odchyleń standardowych od średniej':")
print()
print("  +2.0  ← 2 odchylenia standardowe POWYŻEJ średniej (bardzo wysoka)")
print("  +1.0  ← 1 odchylenie standardowe POWYŻEJ średniej (wysoka)")
print("   0.0  ← DOKŁADNIE średnia")
print("  -1.0  ← 1 odchylenie standardowe PONIŻEJ średniej (niska)")
print("  -2.0  ← 2 odchylenia standardowe PONIŻEJ średniej (bardzo niska)")
print()

# Przykład interpretacji
print("Przykład interpretacji dla pierwszego rekordu:")
for i, col in enumerate(numerical_columns):
    scaled_col = df_scaled.columns[i]
    scaled_val = df_scaled[scaled_col].iloc[idx]
    original_val = df[col].iloc[idx]
    
    if scaled_val > 1:
        interpretation = "znacznie POWYŻEJ średniej"
    elif scaled_val > 0.5:
        interpretation = "nieco POWYŻEJ średniej"
    elif scaled_val > -0.5:
        interpretation = "około średniej"
    elif scaled_val > -1:
        interpretation = "nieco PONIŻEJ średniej"
    else:
        interpretation = "znacznie PONIŻEJ średniej"
    
    print(f"  {col}: {original_val:.1f} → {scaled_val:+.2f} ({interpretation})")
print()

# ============================================================
# KIEDY UŻYWAĆ SKALOWANIA?
# ============================================================
print("=" * 80)
print("KIEDY UŻYWAĆ SKALOWANIA?")
print("=" * 80)
print()

print("WYMAGANE dla algorytmów opartych na ODLEGŁOŚCIACH:")
print("   • Linear Regression (gradient descent)")
print("   • Logistic Regression (gradient descent)")
print("   • Support Vector Machines (SVM)")
print("   • k-Nearest Neighbors (KNN)")
print("   • Neural Networks")
print()

print("NIE WYMAGANE dla algorytmów opartych na DRZEWACH:")
print("   • Decision Tree")
print("   • Random Forest")
print("   • Gradient Boosting (XGBoost, LightGBM)")
print()

print("DLACZEGO?")
print("   Drzewa dzielą dane po wartościach progowych, nie liczą odległości.")
print("   Nie ma znaczenia czy próg to '50000 km' czy '0.5 std'.")
print()

# ============================================================
# REKOMENDACJA DLA NASZYCH MODELI
# ============================================================
print("=" * 80)
print("REKOMENDACJA DLA NASZEGO PROJEKTU")
print("=" * 80)
print()

print("MODEL REGRESJI (Przewidywanie ceny):")
print("  • Algorytm: RandomForestRegressor")
print("  • Skalowanie: NIE POTRZEBNE")
print("  • Powód: Random Forest bazuje na drzewach decyzyjnych")
print()

print("MODEL KLASYFIKACJI (Przewidywanie transmission):")
print("  • Algorytm: LogisticRegression")
print("  • Skalowanie: WYMAGANE")
print("  • Powód: Logistic używa gradient descent (wrażliwy na skalę)")
print()

print("=" * 80)
print("KROK 2.3 ZAKOŃCZONY")
print("=" * 80)
print()