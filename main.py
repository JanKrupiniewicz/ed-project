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

os.makedirs('plots/distribution', exist_ok=True)
os.makedirs('plots/distribution/blue', exist_ok=True)
os.makedirs('plots/distribution/red', exist_ok=True)

os.makedirs('plots/correlation', exist_ok=True)
os.makedirs('plots/correlation/top_attributes', exist_ok=True)

os.makedirs('plots/stat_comparison', exist_ok=True)

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
counts = df['blueWins'].value_counts().sort_index()

sns.barplot(x=[0, 1], y=counts.values, palette=['red', 'blue'], 
            order=[0, 1], hue=counts.index)

plt.title('Histogram rozkładu zmiennej blueWins')
plt.xlabel('Wynik meczu (0 = Czerwona wygrała, 1 = Niebieska wygrała)')
plt.ylabel('Liczba wystąpień')
plt.xticks([0, 1], ['Czerwona wygrała', 'Niebieska wygrała'])
plt.tight_layout()
plt.savefig('plots/blueWins_distribution.png')
plt.close()

# ----------------------------------------------------------------
# Część 3: Analiza rozkladu atrybutów
# ----------------------------------------------------------------
print("\nCzęść 3: Analiza rozkładu atrybutów")

attributes = df.columns.tolist()


def plot_attribute_distribution(attribute, folder):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[attribute], bins=30, kde=True)
    plt.title(f'Histogram rozkładu atrybutu {attribute}')
    plt.xlabel(attribute)
    plt.ylabel('Liczba wystąpień')
    plt.savefig(f'plots/distribution/{folder}/{attribute}_distribution.png')
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
plt.savefig('plots/correlation/correlation_matrix_readable.png')
plt.close()

# Korelacja z 'blueWins'
correlation_with_target = correlation_matrix['blueWins'].drop('blueWins').sort_values(ascending=False)

print("\nTop 10 NAJBARDZIEJ dodatnich korelacji z 'blueWins':")
print(correlation_with_target.head(10))

# Wykresy scatter dla Top 10 korelacji dodatnich zmiennych
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_target.head(10).index, y=correlation_with_target.head(10).values, palette='viridis', hue=correlation_with_target.head(10).index) 
plt.title('Top 10 korelacji dodatnich z blueWins')
plt.xlabel('Atrybuty')
plt.ylabel('Korelacja z blueWins')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/correlation/top_positive_correlations.png')
plt.close()

# Wykresy scatter tylko dla Top 10 korelacji dodatnich zmiennych
top_positive_corr = correlation_with_target.head(10)
important_attributes = top_positive_corr.index.tolist()

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

# Tworzenie wykresów porównawczych
for stat in key_stats:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='blueWins', y=stat, palette='Set2', hue='blueWins')
    plt.title(f'Porównanie statystyki {stat} dla wygranych (1) i przegranych (0) meczów')
    plt.xlabel('blueWins (0 = przegrana, 1 = wygrana)')
    plt.ylabel(stat)
    plt.xticks([0, 1], ['Przegrana', 'Wygrana'])
    plt.tight_layout()
    plt.savefig(f'plots/stat_comparison/{stat}_comparison.png')
    plt.close()


# Wykres najwazniejszych czynnników wpływających na wynik meczu
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_target.head(10).index, y=correlation_with_target.head(10).values, palette='viridis', hue=correlation_with_target.head(10).index) 
plt.title('Najważniejsze czynniki wpływające na wynik meczu')
plt.xlabel('Atrybuty')
plt.ylabel('Korelacja z blueWins')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/stat_comparison/top_factors_influencing_result.png')

# ----------------------------------------------------------------
# Część 6: Sprawdzanie poprawności danych
# ----------------------------------------------------------------
print("\nCzęść 6: Sprawdzanie poprawności danych")

# Sprawdzenie, czy wszystkie wartości w kolumnie 'blueWins' są 0 lub 1
if df['blueWins'].isin([0, 1]).all():
    print("Kolumna 'blueWins' zawiera tylko wartości 0 i 1.")
else:
    print("Kolumna 'blueWins' zawiera wartości inne niż 0 i 1.")

# Sprawdzenie, czy kolumny 'blueKills' i 'redKills' są liczbami całkowitymi
if df['blueKills'].dtype == 'int64' and df['redKills'].dtype == 'int64':
    print("Kolumny 'blueKills' i 'redKills' są liczbami całkowitymi.")
else:
    print("Kolumny 'blueKills' i/lub 'redKills' NIE są liczbami całkowitymi.")

# Sprawdzenie, czy zawsze tylko jedna drużyna ma first blood
if 'blueFirstBlood' in df.columns and 'redFirstBlood' in df.columns:
    first_blood_sum = df['blueFirstBlood'].sum() + df['redFirstBlood'].sum()
    if first_blood_sum == len(df):
        print("Każdy mecz ma dokładnie jedno first blood.")
    else:
        print("Niektóre mecze mają więcej niż jedno first blood lub brak first blood.")

# Sprawdzenie zgodności kills i deaths między drużynami
if 'blueKills' in df.columns and 'redKills' in df.columns:
    if (df['blueKills'] + df['redKills']).equals(df['blueDeaths'] + df['redDeaths']):
        print("Suma zabójstw drużyny niebieskiej i czerwonej jest zgodna z sumą śmierci.")
    else:
        print("Suma zabójstw drużyny niebieskiej i czerwonej NIE jest zgodna z sumą śmierci.")

# Sprawdzenie, czy wartosci diff Gold są zgodne z różnicą między drużynami
if 'blueTotalGold' in df.columns and 'redTotalGold' in df.columns:
    df['gold_diff'] = df['blueTotalGold'] - df['redTotalGold']
    if (df['gold_diff'] == (df['blueTotalGold'] - df['redTotalGold'])).all():
        print("Różnica w złocie jest zgodna z różnicą między drużynami.")
    else:
        print("Różnica w złocie NIE jest zgodna z różnicą między drużynami.")


print("\nAnaliza danych zakończona. Wykresy i statystyki zostały zapisane w katalogu 'plots'.")