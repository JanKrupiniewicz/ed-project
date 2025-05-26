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
plt.savefig('plots/blueWins_distribution.png')
plt.close()

# ----------------------------------------------------------------
# Część 3: Analiza rozkladu atrybutów
# ----------------------------------------------------------------
print("\nCzęść 3: Analiza rozkładu atrybutów")

attributes = df.columns.tolist()

os.makedirs('plots/distribution', exist_ok=True)
os.makedirs('plots/correlation', exist_ok=True)
os.makedirs('plots/correlation_matrix', exist_ok=True)


def plot_attribute_distribution(attribute, folder):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[attribute], bins=30, kde=True)
    plt.title(f'Histogram rozkładu atrybutu {attribute}')
    plt.xlabel(attribute)
    plt.ylabel('Liczba wystąpień')
    plt.savefig(f'plots/{folder}/distribution/{attribute}_distribution.png')
    plt.close()


# Wykresy rozkładu atrybutów dla drużyny niebieskiej
for attr in attributes:
    if 'blue' in attr:
        plot_attribute_distribution(attr, 'blue')

# Wykresy rozkładu atrybutów dla drużyny czerwonej
for attr in attributes:
    if 'red' in attr:
        plot_attribute_distribution(attr, 'red')


# ----------------------------------------------------------------
# Część 4: Analiza korelacji
# ----------------------------------------------------------------
print("\nCzęść 4: Analiza korelacji")

# Obliczenie macierzy korelacji
correlation_matrix = df.corr(numeric_only=True)

# Wizualizacja macierzy korelacji – heatmapa
plt.figure(figsize=(16, 14))
sns.set(font_scale=0.8)
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=0.5, cbar_kws={"shrink": .8})
heatmap.set_title('Macierz korelacji (czytelna)', fontdict={'fontsize': 16})
plt.tight_layout()
plt.savefig('plots/correlation_matrix/correlation_matrix_readable.png')
plt.close()

# Korelacja z 'blueWins'
correlation_with_target = correlation_matrix['blueWins'].drop('blueWins').sort_values(ascending=False)

print("\nTop 10 NAJBARDZIEJ dodatnich korelacji z 'blueWins':")
print(correlation_with_target.head(10))

# Wykres słupkowy – top 10 korelacji dodatnich
top_positive_corr = correlation_with_target.head(10)

plt.figure(figsize=(12, 6))
top_positive_corr.plot(kind='barh', color='green')
plt.title('Top 10 najbardziej dodatnich korelacji z blueWins')
plt.xlabel('Wartość korelacji')
plt.tight_layout()
plt.savefig('plots/correlation_matrix/top_positive_correlations.png')
plt.close()

# Wykresy scatter tylko dla NAJWAŻNIEJSZYCH zmiennych
important_attributes = top_positive_corr.index.tolist()
os.makedirs('plots/correlation/top_attributes', exist_ok=True)

def plot_correlation_with_target(attribute):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df[attribute], y=df['blueWins'])
    plt.title(f'{attribute} vs blueWins')
    plt.xlabel(attribute)
    plt.ylabel('blueWins')
    plt.tight_layout()
    plt.savefig(f'plots/correlation/top_attributes/{attribute}_correlation.png')
    plt.close()

for attr in important_attributes:
    plot_correlation_with_target(attr)


# ----------------------------------------------------------------
# Część 5: Porownanie statystyk dla meczy wygranych i przegranych
# ----------------------------------------------------------------
print("\nCzęść 5: Porównanie statystyk dla meczy wygranych i przegranych")

# Wybór kluczowych statystyk (dla drużyny niebieskiej)
key_stats = [
    'blueKills', 'blueDeaths', 'blueAssists',
    'blueTotalGold', 'blueTotalExperience',
    'blueDragons', 'blueHeralds', 'blueTowersDestroyed',
    'blueWardsPlaced', 'blueWardsDestroyed'
]

# Tworzenie folderu na wykresy
os.makedirs('plots/stat_comparison', exist_ok=True)

# Tworzenie wykresów porównawczych
for stat in key_stats:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='blueWins', y=stat, palette='Set2')
    plt.title(f'Porównanie statystyki {stat} dla wygranych (1) i przegranych (0) meczów')
    plt.xlabel('blueWins (0 = przegrana, 1 = wygrana)')
    plt.ylabel(stat)
    plt.xticks([0, 1], ['Przegrana', 'Wygrana'])
    plt.tight_layout()
    plt.savefig(f'plots/stat_comparison/{stat}_comparison.png')
    plt.close()


# ----------------------------------------------------------------
# Część 6: Analiza wpływu czynników na wygraną
# ----------------------------------------------------------------
print("\nCzęść 6: Analiza wpływu czynników na wygraną")

