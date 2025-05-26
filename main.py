import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------
# CZEŚĆ 1: Wczytanie danych oraz podstawowe operacje
# ----------------------------------------------------------------
print("Część 1: Wczytanie danych oraz podstawowe operacje")

# Wczytanie danych z pliku CSV

df = pd.read_csv('high_diamond_ranked_10min.csv')
print("\nDane zostały wczytane.")

# Pierwsze 5 wierszy DataFrame
print("\nPierwsze 5 wierszy DataFrame:")
print(df.head())

# Podstawowe informacje o DataFrame
print("\nPodstawowe informacje o DataFrame:")
print(df.info())

# Statystyki opisowe
print("\nStatystyki opisowe:")
print(df.describe())



# ----------------------------------------------------------------
# Część 2: Wizualizacja danych
# ----------------------------------------------------------------
print("\nCzęść 2: Wizualizacja danych")

# Dane brakujące
print("\nDane brakujące:")
print(df.isnull().sum())

# Punkty oddalone (outliers)
print("\nPunkty oddalone (outliers):")
outliers = df[(df['blueWins'] < 0) | (df['blueWins'] > 1)]
print(f"Liczba punktów oddalonych: {len(outliers)}")

# Dane niespójne
print("\nDane niespójne:")
inconsistent_data = df[(df['blueWins'] < 0) | (df['blueWins'] > 1)]
print(f"Liczba niespójnych danych: {len(inconsistent_data)}")

# Dane niezrozumiałe
print("\nDane niezrozumiałe:")
unreadable_data = df[df['blueWins'].isnull()]
print(f"Liczba niezrozumiałych danych: {len(unreadable_data)}")


# Histogram rozkładu zmiennej docelowaej 'blueWins'
plt.figure(figsize=(10, 6))
sns.histplot(df['blueWins'], bins=2, kde=False)
plt.title('Histogram rozkładu zmiennej blueWins')
plt.xlabel('blueWins')
plt.ylabel('Liczba wystąpień')
plt.xticks([0, 1], ['Przegrana', 'Wygrana'])
plt.show()


# ----------------------------------------------------------------
# Część 3: Analiza rozkladu atrybutów
# ----------------------------------------------------------------
print("\nCzęść 3: Analiza rozkładu atrybutów")

attributes = df.columns.tolist()

os.makedirs('plots/blue', exist_ok=True)
os.makedirs('plots/red', exist_ok=True)

def plot_attribute_distribution(attribute):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[attribute], bins=30, kde=True)
    plt.title(f'Histogram rozkładu atrybutu {attribute}')
    plt.xlabel(attribute)
    plt.ylabel('Liczba wystąpień')
    plt.savefig(f'plots/blue/{attribute}_distribution.png')
    plt.close()


# Wykresy rozkładu atrybutów dla drużyny niebieskiej
for attr in attributes:
    if 'blue' in attr:
        plot_attribute_distribution(attr)

# Wykresy rozkładu atrybutów dla drużyny czerwonej
for attr in attributes:
    if 'red' in attr:
        plot_attribute_distribution(attr)


# ----------------------------------------------------------------
# Część 4: Analiza korelacji
# ----------------------------------------------------------------
print("\nCzęść 4: Analiza korelacji")

correlation_matrix = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Macierz korelacji')
plt.savefig('plots/correlation_matrix.png')
plt.close()

# Korelacja między atrybutami a zmienną docelową 'blueWins'
correlation_with_target = correlation_matrix['blueWins'].sort_values(ascending=False)
print("\nKorelacja między atrybutami a zmienną docelową 'blueWins':")
print(correlation_with_target)

# Wizualizacja korelacji z wybranymi atrybutami
def plot_correlation_with_target(attribute):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[attribute], y=df['blueWins'])
    plt.title(f'Korelacja między {attribute} a blueWins')
    plt.xlabel(attribute)
    plt.ylabel('blueWins')
    plt.savefig(f'plots/correlation_{attribute}_blueWins.png')
    plt.close()

# Wykresy korelacji z wybranymi atrybutami
for attr in attributes:
    if 'blue' in attr or 'red' in attr:
        plot_correlation_with_target(attr)


# Wizaulizacja macierzy korelacji jako heatmap

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Macierz korelacji')
plt.savefig('plots/correlation_matrix_heatmap.png')
plt.close()

