"""
KROK 2.4: Train/Test Split - Podział danych na zbiór treningowy i testowy
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Ścieżki
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "car_pricing_amjad_zhour.csv"

print("=" * 80)
print("KROK 2.4: TRAIN/TEST SPLIT")
print("=" * 80)
print()

# Wczytanie danych
df = pd.read_csv(DATA_PATH)
print(f"Wczytano {len(df)} rekordów")
print()

# Przygotowanie cech (X) i targetów (y)
print("=" * 80)
print("PRZYGOTOWANIE DANYCH")
print("=" * 80)
print()

# Kolumny cech (features)
feature_columns = ['Make', 'Model', 'Year', 'Engine Size', 'Mileage', 'Fuel Type']

# Dwa różne targety dla dwóch modeli
X = df[feature_columns]
y_price = df['Price']  # Dla regresji
y_transmission = df['Transmission']  # Dla klasyfikacji

print(f"Cechy (X): {len(feature_columns)} kolumn")
print(f"  → {feature_columns}")
print()
print(f"Target 1 (regresja): Price - {len(y_price)} wartości")
print()
print(f"Target 2 (klasyfikacja): Transmission - {len(y_transmission)} wartości")
print(f"  Rozkład: {y_transmission.value_counts().to_dict()}")
print()

# ============================================================
# PODZIAŁ 1: Dla modelu REGRESJI (Price)
# ============================================================
print("=" * 80)
print("PODZIAŁ 1: MODEL REGRESJI (Przewidywanie ceny)")
print("=" * 80)
print()

# Standardowy split 80/20
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_price,
    test_size=0.2,       # 20% na test
    random_state=42      # Powtarzalność
)

print(f"Podział 80/20:")
print(f"  Train: {len(X_train_reg)} rekordów ({len(X_train_reg)/len(X)*100:.1f}%)")
print(f"  Test:  {len(X_test_reg)} rekordów ({len(X_test_reg)/len(X)*100:.1f}%)")
print()

# Statystyki ceny w train i test
print("Statystyki ceny (Price):")
print(f"  TRAIN - Średnia: {y_train_reg.mean():.2f}, Min: {y_train_reg.min():.2f}, Max: {y_train_reg.max():.2f}")
print(f"  TEST  - Średnia: {y_test_reg.mean():.2f}, Min: {y_test_reg.min():.2f}, Max: {y_test_reg.max():.2f}")
print()

diff_mean = abs(y_train_reg.mean() - y_test_reg.mean())
print(f"  Różnica średnich: {diff_mean:.2f} ({diff_mean/y_train_reg.mean()*100:.2f}%)")
if diff_mean/y_train_reg.mean() < 0.05:
    print("Rozkłady są podobne!")
else:
    print("Rozkłady mogą się różnić")
print()

# ============================================================
# PODZIAŁ 2: Dla modelu KLASYFIKACJI (Transmission)
# ============================================================
print("=" * 80)
print("PODZIAŁ 2: MODEL KLASYFIKACJI (Przewidywanie transmission)")
print("=" * 80)
print()

print("DWA SPOSOBY PODZIAŁU:")
print()

# Sposób 1: BEZ stratyfikacji
print("--- Sposób 1: BEZ stratyfikacji ---")
X_train_cls1, X_test_cls1, y_train_cls1, y_test_cls1 = train_test_split(
    X, y_transmission,
    test_size=0.2,
    random_state=42
)

train_dist1 = y_train_cls1.value_counts(normalize=True).sort_index()
test_dist1 = y_test_cls1.value_counts(normalize=True).sort_index()

print(f"  Train rozkład: {dict(y_train_cls1.value_counts())}")
print(f"    → Automatic: {train_dist1.get('Automatic', 0)*100:.1f}%, Manual: {train_dist1.get('Manual', 0)*100:.1f}%")
print(f"  Test rozkład:  {dict(y_test_cls1.value_counts())}")
print(f"    → Automatic: {test_dist1.get('Automatic', 0)*100:.1f}%, Manual: {test_dist1.get('Manual', 0)*100:.1f}%")
print()

# Sposób 2: ZE stratyfikacją
    # Zapewnia, że rozkład klas w train i test jest IDENTYCZNY jak w oryginalnym zbiorze.
    # Ważne dla niezbalansowanych klas (np. 90% Manual, 10% Automatic).

print("--- Sposób 2: ZE stratyfikacją (stratify=y) ---")
X_train_cls2, X_test_cls2, y_train_cls2, y_test_cls2 = train_test_split(
    X, y_transmission,
    test_size=0.2,
    random_state=42,
    stratify=y_transmission  # KLUCZOWY PARAMETR!
)

train_dist2 = y_train_cls2.value_counts(normalize=True).sort_index()
test_dist2 = y_test_cls2.value_counts(normalize=True).sort_index()

print(f"  Train rozkład: {dict(y_train_cls2.value_counts())}")
print(f"    → Automatic: {train_dist2.get('Automatic', 0)*100:.1f}%, Manual: {train_dist2.get('Manual', 0)*100:.1f}%")
print(f"  Test rozkład:  {dict(y_test_cls2.value_counts())}")
print(f"    → Automatic: {test_dist2.get('Automatic', 0)*100:.1f}%, Manual: {test_dist2.get('Manual', 0)*100:.1f}%")
print()


# Rozkład w oryginalnym zbiorze
original_dist = y_transmission.value_counts(normalize=True).sort_index()
print(f"Oryginalny rozkład:")
print(f"  Automatic: {original_dist.get('Automatic', 0)*100:.1f}%, Manual: {original_dist.get('Manual', 0)*100:.1f}%")
print()

# ============================================================
# random_state=42?
# ============================================================
print("=" * 80)
print("DLACZEGO random_state=42?")
print("=" * 80)
print()

print("POWTARZALNOŚĆ EKSPERYMENTÓW:")
print()

# Demonstracja bez random_state
print("BEZ random_state (losowy):")
X_train_a, X_test_a, _, _ = train_test_split(X, y_price, test_size=0.2)
X_train_b, X_test_b, _, _ = train_test_split(X, y_price, test_size=0.2)

# Sprawdzamy czy pierwsze 5 indeksów jest takich samych
same_indices_a = list(X_train_a.index[:5])
same_indices_b = list(X_train_b.index[:5])

print(f"  Podział 1 - pierwsze 5 indeksów train: {same_indices_a}")
print(f"  Podział 2 - pierwsze 5 indeksów train: {same_indices_b}")
print(f"  Czy takie same? {same_indices_a == same_indices_b}")
print()

# Z random_state
print("Z random_state=42:")
X_train_c, X_test_c, _, _ = train_test_split(X, y_price, test_size=0.2, random_state=42)
X_train_d, X_test_d, _, _ = train_test_split(X, y_price, test_size=0.2, random_state=42)

same_indices_c = list(X_train_c.index[:5])
same_indices_d = list(X_train_d.index[:5])

print(f"  Podział 1 - pierwsze 5 indeksów train: {same_indices_c}")
print(f"  Podział 2 - pierwsze 5 indeksów train: {same_indices_d}")
print(f"  Czy takie same? {same_indices_c == same_indices_d}")
print()

print("Z random_state=42:")
print("   • Zawsze ten sam podział")
print("   • Łatwo porównywać eksperymenty")
print("   • Można odtworzyć wyniki")
print()
print("BEZ random_state:")
print("   • Za każdym razem inny podział")
print("   • Trudno porównywać wyniki")
print("   • Nie da się odtworzyć eksperymentu")
print()


# ============================================================
# PODSUMOWANIE I REKOMENDACJE
# ============================================================
print("=" * 80)
print("KROK 2.4 ZAKOŃCZONY")
print("=" * 80)
print()

print("PODSUMOWANIE:")
print()
print("DLA REGRESJI (Przewidywanie ceny):")
print("  • Split: 80/20")
print("  • random_state: 42")
print("  • Stratyfikacja: NIE (nie jest potrzebna dla regresji)")
print(f"  • Train: {len(X_train_reg)} rekordów")
print(f"  • Test:  {len(X_test_reg)} rekordów")
print()

print("DLA KLASYFIKACJI (Przewidywanie transmission):")
print("  • Split: 80/20")
print("  • random_state: 42")
print("  • Stratyfikacja: TAK (stratify=y_transmission)")
print(f"  • Train: {len(X_train_cls2)} rekordów")
print(f"  • Test:  {len(X_test_cls2)} rekordów")
print()
