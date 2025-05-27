# League of Legends - Diamond Ranked Games Analysis

**Projekt z przedmiotu Eksploracja Danych**  
**Etap 1: Analiza danych po 10 minutach gry**

## Autorzy

Jan Krupiniewicz  
Mateusz Fydrych  
Marcin Araśniewicz

## Opis projektu

Projekt oparty jest na analizie danych z gry **League of Legends (LoL)** — jednej z najpopularniejszych gier typu MOBA. Analizowany zbiór zawiera informacje o przebiegu gier rankingowych na wysokim poziomie (DIAMOND I – MASTER), w których rywalizują dwie drużyny: **blue** i **red**.

## Cel analizy

Celem projektu jest **przewidzenie wyniku meczu** (czy drużyna blue wygra – `blueWins = 1`) na podstawie statystyk z **pierwszych 10 minut gry**.

### Kryteria sukcesu:

- Zrozumienie wpływu wczesnych statystyk na wynik meczu
- Identyfikacja najważniejszych cech (features)
- Opracowanie praktycznych wniosków dla graczy
- Analiza różnic między wygrywającymi i przegrywającymi drużynami

## Dane

- Źródło: publiczny zbiór z [Kaggle](https://www.kaggle.com), licencja CC0
- Plik: `high_diamond_ranked_10min.csv`
- Liczba gier: **9 879**
- Klucz główny: `gameId`
- Liczba cech: **38** (po 19 dla każdej drużyny)
- Przykładowe cechy: zabójstwa, złoto, poziom postaci, miniony, smoki, wieże
- Zmienna docelowa: `blueWins` (1 – wygrana drużyny blue, 0 – przegrana)

## Techniczna realizacja

Projekt realizowany w środowisku Python z wykorzystaniem bibliotek do eksploracji i analizy danych (np. pandas, seaborn, scikit-learn).

---

> Projekt realizowany w ramach kursu akademickiego na potrzeby nauki eksploracji danych i modelowania predykcyjnego.
