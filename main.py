import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro

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
os.makedirs('plots/outliers', exist_ok=True)

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

# Dane niespójne
print("\nDane niespójne:")
inconsistent_data = df[(df['blueWins'] < 0) | (df['blueWins'] > 1)]
print(f"Liczba niespójnych danych: {len(inconsistent_data)}")


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

# Wartosc numeryczna ilosci wygranych przez druzyne niebieska meczow
blue_wins_count = df['blueWins'].sum()
print(f"\nLiczba meczów wygranych przez drużynę niebieską: {blue_wins_count}")
# Wartosc numeryczna ilosci przegranych przez druzyne czerwona meczow
red_wins_count = len(df) - blue_wins_count
print(f"Liczba meczów wygranych przez drużynę czerwoną: {red_wins_count}")

# Obliczanie bledu standardowego 
def standard_error(series):
    """Funkcja obliczająca błąd standardowy dla serii danych."""
    return series.std() / np.sqrt(len(series))

# Obliczanie błędu standardowego dla wszystkich kolumn numerycznych
se = df.select_dtypes(include=[np.number]).apply(standard_error)
print("\nBłąd standardowy dla wszystkich kolumn numerycznych:")
print(se)

# Analiza punktow oddalonych (outliers) - metoda z-score

def detect_outliers_z_score(data, threshold=3):
    """Funkcja wykrywająca punkty oddalone (outliers) za pomocą z-score."""
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = [(x - mean) / std_dev for x in data]
    return np.where(np.abs(z_scores) > threshold)

# Wykrywanie punktów oddalonych dla kolumn numerycznych
outliers_dict = {}
for column in df.select_dtypes(include=[np.number]).columns:
    outliers_indices = detect_outliers_z_score(df[column].dropna())
    if len(outliers_indices[0]) > 0:
        outliers_dict[column] = outliers_indices[0]
        print(f"\nPunkty oddalone w kolumnie {column}:")
        print(df[column].iloc[outliers_indices].head())
        print(f"Liczba punktów oddalonych: {len(outliers_indices[0])}")

# Wykresy punktów oddalonych dla kolumn numerycznych
def plot_outliers(column):
    """Funkcja rysująca wykres punktów oddalonych dla danej kolumny."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Wykres punktów oddalonych dla kolumny {column}')
    plt.xlabel(column)
    plt.tight_layout()
    plt.savefig(f'plots/outliers/{column}_outliers.png')
    plt.close()


# Rysowanie wykresów punktów oddalonych dla wszystkich kolumn numerycznych
for column in df.select_dtypes(include=[np.number]).columns:
    plot_outliers(column)

# Srednia liczba punktow oddalonych dla kolumn numerycznych
mean_outliers = {col: len(outliers_dict[col]) for col in outliers_dict}
print("\nŚrednia liczba punktów oddalonych dla kolumn numerycznych:")
print(pd.Series(mean_outliers).mean())

# Procent danych sredniej liczby punktów oddalonych
mean_outliers_percentage = {col: (len(outliers_dict[col]) / len(df)) * 100 for col in outliers_dict}
print("\nProcent danych średniej liczby punktów oddalonych:")
print(pd.Series(mean_outliers_percentage))

# Atrybuty z największą liczbą punktów oddalonych
max_outliers = max(mean_outliers, key=mean_outliers.get)
print(f"\nAtrybut z największą liczbą punktów oddalonych: {max_outliers} ({mean_outliers[max_outliers]} punktów)")

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

    # Dodanie liczbowych wartości (ilości wystąpień) nad słupkami
    counts, bins = np.histogram(df[attribute].dropna(), bins=30)
    for count, left, right in zip(counts, bins[:-1], bins[1:]):
        if count > 0:
            plt.text((left + right) / 2, count, str(count), ha='center', va='bottom', fontsize=8, rotation=90)

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
correlation_matrix = df.corr(numeric_only=True, method='pearson')

print("\nMacierz korelacji:")
print(correlation_matrix)

# Wizualizacja macierzy korelacji
plt.figure(figsize=(16, 14))
sns.set(font_scale=0.8)
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=0.5, cbar_kws={"shrink": .8})
heatmap.set_title('Macierz korelacji metoda Pearsona - naniesione wspolczynniki', fontdict={'fontsize': 16})
plt.tight_layout()
plt.savefig('plots/correlation/correlation_matrix_with_annotations.png')
plt.close()

# Wizualizacja macierzy korelacji bez annotacji
plt.figure(figsize=(16, 14))
sns.set(font_scale=0.8)
heatmap = sns.heatmap(correlation_matrix, cmap='coolwarm', square=True, linewidths=0.5, cbar_kws={"shrink": .8})
heatmap.set_title('Macierz korelacji metoda Pearsona', fontdict={'fontsize': 16})
plt.tight_layout()
plt.savefig('plots/correlation/correlation_matrix.png')
plt.close()


# Korelacja z 'blueWins'
correlation_with_target = correlation_matrix['blueWins'].drop('blueWins').sort_values(ascending=False)

print("\nTop 10 NAJBARDZIEJ dodatnich korelacji z 'blueWins':")
print(correlation_with_target.head(10))

print("\nTop 10 NAJBARDZIEJ ujemnych korelacji z 'blueWins':")
print(correlation_with_target.tail(10))

# Wykresy scatter dla Top 10 korelacji dodatnich zmiennych
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_target.head(10).index, y=correlation_with_target.head(10).values, palette='viridis', hue=correlation_with_target.head(10).index) 
plt.title('Top 10 korelacji dodatnich z blueWins')
plt.xlabel('Atrybuty')
plt.ylabel('Korelacja z blueWins')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/correlation/top_correlations_with_blueWins.png')
plt.close()

# Wykresy scatter dla wszystkich korelacji dodatnich zmiennych
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_target.index, y=correlation_with_target.values, palette='viridis', hue=correlation_with_target.index)
plt.title('Korelacje z blueWins')
plt.xlabel('Atrybuty')
plt.ylabel('Korelacja z blueWins')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('plots/correlation/correlation_with_blueWins.png')
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
plt.close()

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