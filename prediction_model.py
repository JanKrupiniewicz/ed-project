import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from graphviz import Source
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Upewnij się, że masz zainstalowane biblioteki:
# pip install pandas numpy scikit-learn matplotlib seaborn graphviz

# ----------------------------------------------------------------
# CZEŚĆ 1: Wczytanie danych oraz podstawowe operacje
# ----------------------------------------------------------------
print("Część 1: Wczytanie danych oraz podstawowe operacje")

os.makedirs('plots/decision_trees_kfold', exist_ok=True)
os.makedirs('plots/feature_importance', exist_ok=True)

# Wczytanie danych z pliku CSV (założenie, że masz ten plik)
try:
    df = pd.read_csv('high_diamond_ranked_10min.csv')
    print("\nDane zostały wczytane.")
except FileNotFoundError:
    print("Błąd: Plik 'high_diamond_ranked_10min.csv' nie został znaleziony. Upewnij się, że jest w tym samym katalogu co skrypt.")
    exit()

# ----------------------------------------------------------------
# CZEŚĆ 2: Utworzenie modelu predykcyjnego i ocena jego skuteczności
# ----------------------------------------------------------------
print("\nCzęść 2: Utworzenie modelu predykcyjnego i ocena jego skuteczności")

# Przygotowanie danych do modelowania
columns_to_drop = ['gameId', 'blueWins', 'redGoldDiff', 'redExperienceDiff', 'blueDeaths', 'redDeaths', 'redFirstBlood', 'redDragons', 'redHeralds', 'redEliteMonsters']
X = df.drop(columns=columns_to_drop)
X = X.select_dtypes(include=[np.number])
y = df['blueWins']

print(f"\nDane zostały przygotowane. Liczba próbek: {X.shape[0]}, liczba cech: {X.shape[1]}")


scaler = StandardScaler()

feature_importances_per_fold = []

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_num = 1
optimal_max_depth = 4

print(f"\nRozpoczynanie 10-krotnej kroswalidacji z max_depth={optimal_max_depth}...")

for train_index, test_index in kf.split(X, y):
    print(f"\nTrening i ocena dla Fold {fold_num}...")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Trening modelu DecisionTreeClassifier z class_weight='balanced' 
    model = DecisionTreeClassifier(max_depth=optimal_max_depth, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    # Ocena modelu (opcjonalnie, bo użytkownik już to ma)
    y_pred = model.predict(X_test_scaled)
    print(f"Classification Report for Fold {fold_num}:\n{classification_report(y_test, y_pred)}")

    # Wizualizacja drzewa decyzyjnego dla każdego folda
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=X.columns,
        class_names=['Red', 'Blue'],
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = Source(dot_data)
    graph.render(f'plots/decision_trees_kfold/decision_tree_fold_{fold_num}', format='png', cleanup=True)
    print(f"Drzewo decyzyjne dla Fold {fold_num} zapisane jako 'plots/decision_trees_kfold/decision_tree_fold_{fold_num}.png'")

    # Zbieranie ważności cech
    feature_importances_per_fold.append(model.feature_importances_)

    fold_num += 1

print("\nGenerowanie słupkowego wykresu uśrednionej istotności cech...")

# Obliczanie uśrednionej ważności cech
# Konwersja listy array'ów ważności cech na DataFrame, a następnie obliczenie średniej
avg_feature_importances = pd.DataFrame(feature_importances_per_fold, columns=X.columns).mean().sort_values(ascending=False)

# Słupkowy wykres uśrednionej istotności drzew decyzyjnych
plt.figure(figsize=(12, 8))
sns.barplot(x=avg_feature_importances.values, y=avg_feature_importances.index)
plt.title('Słupkowy wykres uśrednionej istotności drzew decyzyjnych', fontsize=16)
plt.xlabel('Uśredniona istotność', fontsize=14)
plt.ylabel('Cechy', fontsize=14)
plt.tight_layout()
plt.savefig('plots/feature_importance/average_feature_importance_bar_chart.png')
plt.close()
print("Słupkowy wykres uśrednionej istotności drzew decyzyjnych zapisany jako 'plots/feature_importance/average_feature_importance_bar_chart.png'")

# ----------------------------------------------------------------
# Część 3: Eksperymenty z modelem i zbiorem danych
# ----------------------------------------------------------------
print("\nCzęść 3: Eksperymenty z modelem i zbiorem danych")

# Wybieranie roznych podzbiorów atrybutów
top_one_features = avg_feature_importances.head(1).index.tolist()
top_three_features = avg_feature_importances.head(3).index.tolist()
top_four_features = avg_feature_importances.head(4).index.tolist()
top_five_features = avg_feature_importances.head(5).index.tolist()


for features_subset in [top_one_features, top_three_features, top_four_features, top_five_features]:
    subset_name = '_'.join(features_subset)
    print(f"\nTrening modelu z podzbiorem cech: {subset_name}")

    # Przygotowanie danych z wybranym podzbiorem cech
    X_subset = X[features_subset]

    # Skalowanie cech
    X_subset_scaled = scaler.fit_transform(X_subset)

    # Trening modelu DecisionTreeClassifier
    model_subset = DecisionTreeClassifier(max_depth=optimal_max_depth, random_state=42, class_weight='balanced')
    model_subset.fit(X_subset_scaled, y)

    accuracy = model.score(X_test_scaled, y_test)
    sensitivity = np.mean(y_pred[y_test == 1] == 1)
    specificity = np.mean(y_pred[y_test == 0] == 0)
    accuracy_std = np.std(y_pred == y_test)
    sensitivity_std = np.std(y_pred[y_test == 1] == 1)
    specificity_std = np.std(y_pred[y_test == 0] == 0)

    # Dokładność modelu +- odchylenie standardowe
    print(f"Dokładność (średnia +- odchylenie standardowe): {accuracy:.4f} ± {accuracy_std:.4f}")

    # Czulość modelu +- odchylenie standardowe
    print(f"Czułość (średnia +- odchylenie standardowe): {sensitivity:.4f} ± {sensitivity_std:.4f}")

    # Swoistość (średnia +-odchylenie standardowe)
    print(f"Swoistość (średnia +- odchylenie standardowe): {specificity:.4f} ± {specificity_std:.4f}")


# ------------------------------------------------------------------
# Część 4: Dobór innych parametrów pracy algorytmu (Hyperparameter Tuning for Decision Tree)
# ------------------------------------------------------------------
print("\nCzęść 4: Dobór innych parametrów pracy algorytmu")

# Ustawienia dla kroswalidacji
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# Lista do przechowywania wyników
results = []

# Zakres parametrów do przeszukania
max_depth_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
min_samples_split_values = [2, 5, 10, 20]
min_samples_leaf_values = [1, 2, 5, 10]
criterion_values = ['gini', 'entropy']

# Przeszukiwanie losowe przez 100 iteracji

random.seed(42)
for _ in range(100):
    max_depth = random.choice(max_depth_values)
    min_samples_split = random.choice(min_samples_split_values)
    min_samples_leaf = random.choice(min_samples_leaf_values)
    criterion = random.choice(criterion_values)

    fold_accuracies = []
    fold_sensitivities = []
    fold_specificities = []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Trening modelu DecisionTreeClassifier z losowymi parametrami
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train_scaled, y_train)

        # Ocena modelu
        y_pred = model.predict(X_test_scaled)
        accuracy = np.mean(y_pred == y_test)
        sensitivity = np.mean(y_pred[y_test == 1] == 1)
        specificity = np.mean(y_pred[y_test == 0] == 0)

        fold_accuracies.append(accuracy)
        fold_sensitivities.append(sensitivity)
        fold_specificities.append(specificity)

    # Uśrednianie wyników dla danego zestawu parametrów
    avg_accuracy = np.mean(fold_accuracies)
    avg_sensitivity = np.mean(fold_sensitivities)
    avg_specificity = np.mean(fold_specificities)

    results.append({
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'criterion': criterion,
        'accuracy': avg_accuracy,
        'sensitivity': avg_sensitivity,
        'specificity': avg_specificity
    })

# Wartosci uzyskane z przeszukiwania
results_df = pd.DataFrame(results)
# Sortowanie wyników według dokładności
results_df = results_df.sort_values(by='accuracy', ascending=False)
print("\nNajlepsze wyniki z przeszukiwania losowego:")
print(results_df.head(10))
# Zapis wyników do pliku CSV
results_df.to_csv('hyperparameter_tuning_results.csv', index=False)

# Wykres porównawczy wyników
plt.figure(figsize=(12, 8))
sns.barplot(x='accuracy', y='max_depth', data=results_df)
plt.title('Porównanie dokładności modeli dla różnych parametrów', fontsize=16)
plt.xlabel('Dokładność', fontsize=14)
plt.ylabel('Maksymalna głębokość drzewa', fontsize=14)
plt.tight_layout()
plt.savefig('plots/hyperparameter_tuning_accuracy_comparison.png')
plt.close()

# ----------------------------------------------------------------
# Część 5: Sprawdzenie algorytmów alternatywnych
# ----------------------------------------------------------------
print("\nCzęść 5: Sprawdzenie algorytmów alternatywnych")

# Ocena RandomForestClassifier 
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_classifier.fit(X_train_scaled, y_train)
y_pred_rf = rf_classifier.predict(X_test_scaled)

print("\nOcena modelu RandomForestClassifier:")
print(classification_report(y_test, y_pred_rf))

# Ocena modelu LogisticRegression
lr_classifier = LogisticRegression(max_iter=5000, random_state=42, class_weight='balanced')
lr_classifier.fit(X_train_scaled, y_train)
y_pred_lr = lr_classifier.predict(X_test_scaled)
print("\nOcena modelu LogisticRegression:")
print(classification_report(y_test, y_pred_lr))

# Ocena modelu KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_scaled, y_train)
y_pred_knn = knn_classifier.predict(X_test_scaled)

print("\nOcena modelu KNeighborsClassifier:")
print(classification_report(y_test, y_pred_knn))

# Podsumowanie wyników wszystkich modeli
print("\nPodsumowanie wyników wszystkich modeli:")
models = {
    'Decision Tree': model,
    'Random Forest': rf_classifier,
    'Logistic Regression': lr_classifier,
    'K-Nearest Neighbors': knn_classifier
}

for model_name, model_instance in models.items():
    y_pred = model_instance.predict(X_test_scaled)
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))

# Wykres porównawczy dokładności modeli
model_names = list(models.keys())
accuracies = [model.score(X_test_scaled, y_test) for model in models.values()]
plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=accuracies)
plt.title('Porównanie dokładności modeli', fontsize=16)
plt.xlabel('Modele', fontsize=14)
plt.ylabel('Dokładność', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/model_comparison_accuracy.png')
plt.close()

# Wykres porównawczy czułości modeli
sensitivities = [np.mean(y_pred[y_test == 1] == 1) for y_pred in [model.predict(X_test_scaled) for model in models.values()]]
plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=sensitivities)
plt.title('Porównanie czułości modeli', fontsize=16)
plt.xlabel('Modele', fontsize=14)
plt.ylabel('Czułość', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/model_comparison_sensitivity.png')
plt.close()

# Wykres porównawczy swoistości modeli
specificities = [np.mean(y_pred[y_test == 0] == 0) for y_pred in [model.predict(X_test_scaled) for model in models.values()]]
plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=specificities)
plt.title('Porównanie swoistości modeli', fontsize=16)
plt.xlabel('Modele', fontsize=14)
plt.ylabel('Swoistość', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/model_comparison_specificity.png')
plt.close()

# Testowanie modeli metodą 10 krotnej walidacji krzyżowej
from sklearn.model_selection import cross_val_score
for model_name, model_instance in models.items():
    cv_scores = cross_val_score(model_instance, X, y, cv=kf, scoring='accuracy')
    print(f"\n{model_name} - Średnia dokładność z 10-krotnej walidacji krzyżowej: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")


# ----------------------------------------------------------------
# Część 6: Podsumowanie i zakończenie
# ----------------------------------------------------------------