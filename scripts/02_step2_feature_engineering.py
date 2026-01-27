"""
KROK 2.2: Feature Engineering - Tworzenie nowych cech
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Ścieżki
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "car_pricing_amjad_zhour.csv"

print("=" * 80)
print("KROK 2.2: FEATURE ENGINEERING")
print("=" * 80)
print()

# Wczytanie danych
df = pd.read_csv(DATA_PATH)
print(f"Wczytano {len(df)} rekordów")
print()

# Aktualna data (dla obliczeń wieku)
CURRENT_YEAR = 2026

print("=" * 80)
print("DANE PRZED FEATURE ENGINEERING")
print("=" * 80)
print()
print("Przykładowe 5 wierszy (oryginalne cechy numeryczne):")
print(df[['Year', 'Mileage', 'Price']].head())
print()

# Korelacja z ceną PRZED
print("Korelacja z ceną (PRZED Feature Engineering):")
correlation_before = df[['Year', 'Mileage', 'Price']].corr()['Price'].sort_values(ascending=False)
print(correlation_before)
print()

# ============================================================
# FEATURE 1: Car Age (Wiek samochodu)
# ============================================================
print("=" * 80)
print("FEATURE 1: Car_Age = 2026 - Year")
print("=" * 80)
print()

df['Car_Age'] = CURRENT_YEAR - df['Year']

print("Przykłady transformacji:")
print(df[['Year', 'Car_Age']].head(10))
print()

print(f"Statystyki Car_Age:")
print(f"  Min: {df['Car_Age'].min()} lat (najnowsze auto)")
print(f"  Max: {df['Car_Age'].max()} lat (najstarsze auto)")
print(f"  Średnia: {df['Car_Age'].mean():.1f} lat")
print()

# Korelacja Car_Age z ceną
car_age_corr = df[['Car_Age', 'Price']].corr()['Price']['Car_Age']
year_corr = df[['Year', 'Price']].corr()['Price']['Year']

print("Porównanie korelacji z ceną:")
print(f"  Year:     {year_corr:+.4f}")
print(f"  Car_Age:  {car_age_corr:+.4f}")
print()

if abs(car_age_corr) > abs(year_corr):
    print("Car_Age ma SILNIEJSZĄ korelację z ceną!")
else:
    print("Year ma silniejszą korelację")
print()

# ============================================================
# FEATURE 2: Mileage per Year (opcjonalnie)
# ============================================================
print("=" * 80)
print("FEATURE 2: Mileage_per_Year = Mileage / Car_Age")
print("=" * 80)
print()

# Uwaga: samochody z Car_Age = 0 (rok 2026) mogą mieć problem z dzieleniem
# Zabezpieczenie: jeśli Car_Age = 0, użyj 1
df['Mileage_per_Year'] = df['Mileage'] / df['Car_Age'].replace(0, 1)

print("Przykłady transformacji:")
print(df[['Year', 'Car_Age', 'Mileage', 'Mileage_per_Year']].head(10))
print()

print(f"Statystyki Mileage_per_Year:")
print(f"  Min: {df['Mileage_per_Year'].min():.0f} km/rok")
print(f"  Max: {df['Mileage_per_Year'].max():.0f} km/rok")
print(f"  Średnia: {df['Mileage_per_Year'].mean():.0f} km/rok")
print(f"  Mediana: {df['Mileage_per_Year'].median():.0f} km/rok")
print()

# Interpretacja
print("INTERPRETACJA:")
print("  • Mileage_per_Year pokazuje 'intensywność użytkowania'")
print("  • Niski km/rok = mało używane (może więcej warte)")
print("  • Wysoki km/rok = intensywnie użytkowane (może mniej warte)")
print()

# Korelacja z ceną
mileage_per_year_corr = df[['Mileage_per_Year', 'Price']].corr()['Price']['Mileage_per_Year']
mileage_corr = df[['Mileage', 'Price']].corr()['Price']['Mileage']

print("Porównanie korelacji z ceną:")
print(f"  Mileage:          {mileage_corr:+.4f}")
print(f"  Mileage_per_Year: {mileage_per_year_corr:+.4f}")
print()

if abs(mileage_per_year_corr) > abs(mileage_corr):
    print("Mileage_per_Year ma SILNIEJSZĄ korelację!")
else:
    print("Mileage ma silniejszą korelację")
print()

# ============================================================
# PODSUMOWANIE - Wszystkie korelacje
# ============================================================
print("=" * 80)
print("PODSUMOWANIE KORELACJI Z CENĄ")
print("=" * 80)
print()

features_for_correlation = ['Price', 'Year', 'Car_Age', 'Mileage', 'Mileage_per_Year', 'Engine Size']
correlation_final = df[features_for_correlation].corr()['Price'].sort_values(ascending=False)

print("Ranking cech według siły korelacji z ceną:")
for i, (feature, corr_value) in enumerate(correlation_final.items(), 1):
    if feature != 'Price':
        icon = "" if corr_value > 0 else ""
        print(f"  {i}. {icon} {feature:20s}: {corr_value:+.4f}")
print()

# ============================================================
# REKOMENDACJA
# ============================================================
print("=" * 80)
print("REKOMENDACJA")
print("=" * 80)
print()

# Porównanie Year vs Car_Age
if abs(car_age_corr) > abs(year_corr):
    print("UŻYWAJ: Car_Age zamiast Year")
    print(f"   Powód: Silniejsza korelacja ({abs(car_age_corr):.4f} vs {abs(year_corr):.4f})")
else:
    print("UŻYWAJ: Year (Car_Age nie poprawia)")

print()

# Porównanie Mileage vs Mileage_per_Year
if abs(mileage_per_year_corr) > abs(mileage_corr):
    print("UŻYWAJ: Mileage_per_Year zamiast Mileage")
    print(f"   Powód: Silniejsza korelacja ({abs(mileage_per_year_corr):.4f} vs {abs(mileage_corr):.4f})")
else:
    print("UŻYWAJ: Mileage (Mileage_per_Year nie poprawia)")

print()
print("=" * 80)
print("KROK 2.2 ZAKOŃCZONY")
print("=" * 80)
print()
print("Użyjemy Year + Mileage (oryginalne) - daje największą korelację.")

print()
