"""
Skrypt eksploracji danych - Analiza zbioru danych samochodów
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Ścieżki
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "car_pricing_amjad_zhour.csv"

print("=" * 80)
print("EKSPLORACJA DANYCH - Analiza zbioru samochodów")
print("=" * 80)
print()

# 1. Wczytanie danych
print("Wczytywanie danych...")
df = pd.read_csv(DATA_PATH)
print(f"Wczytano {len(df)} rekordów")
print()

# 2. Podstawowe informacje
print("=" * 80)
print("PODSTAWOWE INFORMACJE O ZBIORZE DANYCH")
print("=" * 80)
print(f"Liczba rekordów: {df.shape[0]}")
print(f"Liczba kolumn: {df.shape[1]}")
print(f"Kolumny: {list(df.columns)}")
print()

# 3. Pierwsze rekordy
print("=" * 80)
print("PRZYKŁADOWE DANE (pierwsze 5 rekordów)")
print("=" * 80)
print(df.head())
print()

# 4. Informacje o typach danych
print("=" * 80)
print("TYPY DANYCH I BRAKI")
print("=" * 80)
print(df.info())
print()

# 5. Braki danych (missing values)
print("=" * 80)
print("BRAKI DANYCH (Missing Values)")
print("=" * 80)
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Kolumna': missing.index,
    'Braki': missing.values,
    'Procent': missing_percent.values
})
print(missing_df)
print()

if missing.sum() == 0:
    print("BRAK braków danych - zbiór jest kompletny!")
else:
    print(f"UWAGA: Znaleziono {missing.sum()} braków danych")
print()

# 6. Statystyki dla kolumn numerycznych
print("=" * 80)
print("STATYSTYKI KOLUMN NUMERYCZNYCH")
print("=" * 80)
print(df.describe())
print()

# 7. Analiza kolumn kategorycznych
print("=" * 80)
print("KOLUMNY KATEGORYCZNE - ROZKŁAD WARTOŚCI")
print("=" * 80)

categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"\n--- {col} ---")
    print(df[col].value_counts())
    print(f"Liczba unikalnych wartości: {df[col].nunique()}")

# 8. Szczegółowa analiza kolumn numerycznych
print("\n" + "=" * 80)
print("ANALIZA KOLUMN NUMERYCZNYCH")
print("=" * 80)

numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    print(f"\n--- {col} ---")
    print(f"Min: {df[col].min()}")
    print(f"Max: {df[col].max()}")
    print(f"Średnia: {df[col].mean():.2f}")
    print(f"Mediana: {df[col].median():.2f}")
    print(f"Odchylenie std: {df[col].std():.2f}")
    
    # Sprawdzenie outlierów (wartości poza 3 * std)
    mean = df[col].mean()
    std = df[col].std()
    outliers = df[(df[col] < mean - 3*std) | (df[col] > mean + 3*std)]
    if len(outliers) > 0:
        print(f"Wykryto {len(outliers)} potencjalnych outlierów")

# 9. Analiza korelacji dla zadań predykcyjnych
print("\n" + "=" * 80)
print("KORELACJA Z CENĄ (dla regresji)")
print("=" * 80)
correlation_with_price = df[numeric_columns].corr()['Price'].sort_values(ascending=False)
print(correlation_with_price)
print()

# 10. Analiza zakresu lat
print("=" * 80)
print("ANALIZA ROKU PRODUKCJI")
print("=" * 80)
print(f"Najstarszy samochód: {df['Year'].min()}")
print(f"Najnowszy samochód: {df['Year'].max()}")
print(f"Zakres: {df['Year'].max() - df['Year'].min()} lat")
print()

# 11. Podsumowanie i wnioski
print("=" * 80)
print("PODSUMOWANIE I POTENCJALNE PROBLEMY")
print("=" * 80)

issues = []

# Sprawdzenie braków
if missing.sum() > 0:
    issues.append(f"{missing.sum()} braków danych")
else:
    issues.append("Brak braków danych")

# Sprawdzenie duplikatów
duplicates = df.duplicated().sum()
if duplicates > 0:
    issues.append(f"{duplicates} zduplikowanych rekordów")
else:
    issues.append("Brak duplikatów")

# Sprawdzenie wartości
if (df['Price'] <= 0).any():
    issues.append(f"Ujemne/zerowe ceny: {(df['Price'] <= 0).sum()}")
else:
    issues.append("Wszystkie ceny są dodatnie")

if (df['Mileage'] < 0).any():
    issues.append(f"Ujemny przebieg: {(df['Mileage'] < 0).sum()}")
else:
    issues.append("Wszystkie przebiegi są nieujemne")

for issue in issues:
    print(issue)

print()
print("=" * 80)
print("EKSPLORACJA DANYCH ZAKOŃCZONA")
print("=" * 80)
print()
print("DANE SĄ GOTOWE DO:")
print("  1. Model Regresji: Przewidywanie CENY na podstawie cech")
print("  2. Model Klasyfikacji: Przewidywanie TRANSMISSION (Manual/Automatic)")
print()
