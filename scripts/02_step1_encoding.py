"""
KROK 2.1: One-Hot Encoding - zmienne kategoryczne na format numeryczny
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

# Ścieżki
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "car_pricing_amjad_zhour.csv"

print("=" * 80)
print("KROK 2.1: ONE-HOT ENCODING")
print("=" * 80)
print()

# Wczytanie danych
df = pd.read_csv(DATA_PATH)
print(f"Wczytano {len(df)} rekordów")
print()

# Zmienne kategoryczne do zakodowania
categorical_columns = ['Make', 'Model', 'Fuel Type']

print("=" * 80)
print("DANE PRZED ENCODING")
print("=" * 80)
print()
print("Przykładowe 5 wierszy (tylko kolumny kategoryczne):")
print(df[categorical_columns].head())
print()

# Sprawdźmy unikalne wartości
print("Unikalne wartości w każdej kolumnie:")
for col in categorical_columns:
    unique_vals = df[col].unique()
    print(f"\n{col}: {len(unique_vals)} wartości")
    print(f"  → {list(unique_vals)}")
print()

# Zastosowanie One-Hot Encoding
print("=" * 80)
print("ZASTOSOWANIE ONE-HOT ENCODING")
print("=" * 80)
print()

encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' zapobiega multikolinearności
encoded_data = encoder.fit_transform(df[categorical_columns])

# Nazwy kolumn po encoding
feature_names = encoder.get_feature_names_out(categorical_columns)
print(f"Utworzono {len(feature_names)} nowych kolumn (drop='first' redukuje o 3)")
print()

# Utworzenie DataFrame z zakodowanymi danymi
df_encoded = pd.DataFrame(encoded_data, columns=feature_names)

print("=" * 80)
print("DANE PO ENCODING")
print("=" * 80)
print()
print("Nowe kolumny (wszystkie):")
for i, col_name in enumerate(feature_names, 1):
    print(f"  {i:2d}. {col_name}")
print()

print("Przykładowe 5 wierszy (zakodowane dane):")
print(df_encoded.head())
print()

# Połączenie z oryginalnymi danymi numerycznymi
numerical_columns = ['Year', 'Engine Size', 'Mileage', 'Price']
df_full = pd.concat([
    df[numerical_columns].reset_index(drop=True),
    df_encoded
], axis=1)

print("=" * 80)
print("POŁĄCZENIE: Dane numeryczne + One-Hot Encoded")
print("=" * 80)
print()
print(f"Liczba kolumn PRZED: {len(categorical_columns) + len(numerical_columns)} (4 numeryczne + 3 kategoryczne)")
print(f"Liczba kolumn PO: {len(df_full.columns)} (4 numeryczne + {len(feature_names)} one-hot)")
print()
print("Przykład kompletnego rekordu:")
print(df_full.head(3))
print()

# Przykład interpretacji
print("=" * 80)
print("JAK CZYTAĆ ONE-HOT ENCODING?")
print("=" * 80)
print()
print("Przykład - pierwszy wiersz z oryginalnych danych:")
print(f"  Make: {df['Make'].iloc[0]}")
print(f"  Model: {df['Model'].iloc[0]}")
print(f"  Fuel Type: {df['Fuel Type'].iloc[0]}")
print()
print("Po encoding (pierwszy wiersz):")
for col in feature_names:
    value = df_encoded[col].iloc[0]
    if value == 1.0:
        print(f"  {col} = {value} ← TO JEST TA KATEGORIA")
print()
print("  drop='first': Usuwamy pierwszą kategorię z każdej grupy")
print("   (zapobiega multikolinearności - sytuacji gdzie suma kolumn = 1)")
print()

print("=" * 80)
print("KROK 2.1 ZAKOŃCZONY")
print("=" * 80)
print()
print("PODSUMOWANIE:")
print(f"  • Zakodowano {len(categorical_columns)} zmienne kategoryczne")
print(f"  • Utworzono {len(feature_names)} kolumn binarnych")
print(f"  • Każda kolumna ma wartość 0 lub 1")
print(f"  • Model teraz 'rozumie' kategorie jako liczby")
print()
