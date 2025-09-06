# Narzędzie Analizy Podobieństwa Dokumentów - Dokumentacja Użytkownika

## Opis Projektu

To narzędzie zapewnia automatyczną analizę dokumentów tekstowych (książek, artykułów, raportów, prac naukowych) w celu wykrywania powtarzających się wzorców treści. Wykorzystuje zaawansowane techniki NLP do grupowania podobnych akapitów w klastry i generowania zwięzłych podsumowań dla każdej grupy.

### Główne Korzyści
- **Oszczędność Czasu**: Automatyczne identyfikowanie nadmiarowej treści, która ręcznie zajęłaby godziny
- **Poprawa Jakości**: Pomaga autorom i redaktorom identyfikować niezamierzone powtórzenia
- **Optymalizacja Treści**: Umożliwia tworzenie bardziej zwięzłych dokumentów poprzez wyróżnienie duplikujących się informacji
- **Integralność Akademicka**: Przydatne do wykrywania autoplagiatów w pracach naukowych

## Przypadki Użycia

1. **Prace Naukowe**: Identyfikacja powtarzających się wyjaśnień lub definicji
2. **Dokumentacja Techniczna**: Znajdowanie duplikujących się instrukcji lub specyfikacji
3. **Raporty Biznesowe**: Wykrywanie nadmiarowych analiz lub wniosków
4. **Książki i Artykuły**: Lokalizowanie powtarzających się tematów lub przykładów
5. **Dokumenty Prawne**: Identyfikacja duplikujących się klauzul lub warunków

## Szybki Start

### Wymagania Wstępne
```bash
# Instalacja wymaganych pakietów
pip install openai scikit-learn sentence-transformers numpy pandas python-docx PyMuPDF
```

### Konfiguracja Środowiska
```bash
# Windows
setx OPENAI_API_KEY "twój_klucz_api_tutaj"

# macOS
export OPENAI_API_KEY="twój_klucz_api_tutaj"
```

### Struktura Projektu

**Windows:**
```
C:\Users\[TwojeImię]\Documents\DocumentAnalysis\Projects\twój_projekt\
├── dokument.pdf              # Twój dokument źródłowy
├── similarity_tool.py         # Główny silnik analizy
├── run_document.py           # Skrypt główny
├── analyze_results_v2.py     # Klasyfikacja typów treści
├── analyze_results_v3.py     # Podsumowania AI
├── extract_clusters.py       # Narzędzie ekstrakcji klastrów
├── extract_examples.py       # Selektor reprezentatywnych przykładów
├── results_doc.csv           # Surowe wyniki analizy
├── clusters/                 # Wyodrębnione pliki klastrów
├── clusters_v2/              # Sklasyfikowane klastry
└── clusters_v3/              # Podsumowane klastry
```

**macOS:**
```
~/Documents/DocumentAnalysis/Projects/twój_projekt/
├── dokument.pdf              # Twój dokument źródłowy
├── similarity_tool.py         # Główny silnik analizy
├── run_document.py           # Skrypt główny
├── analyze_results_v2.py     # Klasyfikacja typów treści
├── analyze_results_v3.py     # Podsumowania AI
├── extract_clusters.py       # Narzędzie ekstrakcji klastrów
├── extract_examples.py       # Selektor reprezentatywnych przykładów
├── results_doc.csv           # Surowe wyniki analizy
├── clusters/                 # Wyodrębnione pliki klastrów
├── clusters_v2/              # Sklasyfikowane klastry
└── clusters_v3/              # Podsumowane klastry
```

## Przepływ Pracy

### Krok 1: Analiza Początkowa
```python
# Edytuj run_document.py
PROJECT_NAME = "moja_praca"
PROJECT_FILE = "dokument.pdf"

# Uruchom analizę
python run_document.py --project moja_praca --file dokument.pdf
```

### Krok 2: Klasyfikacja
```python
# Klasyfikuj typy treści (spis treści, tabele, treść)
python analyze_results_v2.py --project moja_praca
```

### Krok 3: Podsumowania AI
```python
# Generuj podsumowania klastrów za pomocą GPT-4
python analyze_results_v3.py
```

### Krok 4: Wyodrębnianie Przykładów
```python
# Uzyskaj reprezentatywne próbki z każdego klastra
python extract_examples.py --project moja_praca
```

## Opcje Konfiguracji

### Wybór Modelu
- **gpt-4o**: Najlepsza jakość, szczególnie dla języka polskiego
- **gpt-4o-mini**: Budżetowa opcja do testowania

### Parametry Klastrowania
- **eps** (0.3): Próg podobieństwa - niższy = bardziej rygorystyczne dopasowanie
- **min_samples** (2): Minimalny rozmiar klastra
- **chunk_size** (20): Akapity przetwarzane na partię

## Interpretacja Wyników

### Kategorie Podobieństwa
- **<10%**: Niskie podobieństwo - prawdopodobnie unikalna treść
- **10-25%**: Pewne podobieństwo - powiązane tematy
- **25-50%**: Podobne - nakładające się koncepcje
- **50-75%**: Bardzo podobne - prawdopodobnie powtarzające się
- **>75%**: Niemal identyczne - silne powielanie

### Typy Klastrów
- **UNIQUE**: Pojedyncze akapity bez duplikatów
- **SIMILAR-XX**: Grupy powiązanej/powielonej treści
- **TOC**: Wpisy spisu treści
- **Tabela/Case**: Tabele lub studia przypadków

## Najlepsze Praktyki

1. **Przetwarzanie Wstępne**: Wyczyść dokument z nagłówków/stopek przed analizą
2. **Język**: Narzędzie działa z mieszaną treścią polsko-angielską
3. **Rozmiar Pliku**: Dla dokumentów >100 stron rozważ podział na rozdziały
4. **Użycie API**: Monitoruj użycie API OpenAI, aby kontrolować koszty
5. **Tryb Offline**: Narzędzie działa bez API, ale nie generuje podsumowań

## Zaawansowane Użycie

### Niestandardowe Progi Podobieństwa
```python
# W similarity_tool.py, dostosuj parametry klastrowania:
labels = cluster_paragraphs(embeddings, eps=0.25, min_samples=3)
```

### Przetwarzanie Wsadowe Wielu Dokumentów
```python
import glob
for file in glob.glob("*.pdf"):
    process_document(file, model_name="gpt-4o")
```

### Eksport do Różnych Formatów
```python
# CSV do analizy danych
df.to_csv("wyniki.csv", encoding="utf-8")

# JSON do aplikacji webowych
df.to_json("wyniki.json", orient="records")

# Excel dla użytkowników biznesowych
df.to_excel("wyniki.xlsx", index=False)
```

## Objaśnienie Plików Wyjściowych

| Plik | Opis | Przypadek Użycia |
|------|------|------------------|
| output.docx | Dokument z adnotacjami i etykietami klastrów | Przegląd duplikatów w kontekście |
| results_doc.csv | Surowe mapowanie akapit-klaster | Analiza danych i filtrowanie |
| cluster_summary_v2.csv | Sklasyfikowane klastry według typu | Kategoryzacja treści |
| cluster_summary_v3.csv | Podsumowania generowane przez AI | Szybki przegląd duplikatów |
| cluster_examples.csv | Przykładowe akapity na klaster | Weryfikacja jakości |

## Rozwiązywanie Problemów

### Częste Problemy

1. **"Brak klucza API"**: Ustaw zmienną środowiskową OPENAI_API_KEY
2. **"UnicodeDecodeError"**: Zapewnij kodowanie UTF-8 w operacjach na plikach
3. **"Błąd pamięci"**: Zmniejsz chunk_size dla dużych dokumentów
4. **"Nie znaleziono klastrów"**: Dostosuj parametr eps (spróbuj 0.4 lub 0.5)

### Wskazówki Dotyczące Wydajności

- Przetwarzaj dokumenty w sekcjach dla lepszego zarządzania pamięcią
- Użyj gpt-4o-mini do początkowych testów, aby zmniejszyć koszty
- Cachuj embeddingi dla powtarzanych analiz
- Rozważ lokalne modele LLM dla wrażliwych dokumentów

## Szczegółowe Instrukcje Krok po Krok

### Przygotowanie Środowiska

1. **Instalacja Python** (wymagana wersja 3.8+):
   - Windows: Pobierz z [python.org](https://python.org)
   - macOS: Użyj Homebrew: `brew install python`

2. **Instalacja Zależności**:
   ```bash
   pip install openai scikit-learn sentence-transformers numpy pandas python-docx PyMuPDF python-dotenv
   ```

3. **Konfiguracja Klucza API**:
   - Uzyskaj klucz API z [OpenAI](https://platform.openai.com/api-keys)
   - Windows: `setx OPENAI_API_KEY "twój_klucz"`
   - macOS: Dodaj do `~/.bash_profile` lub `~/.zshrc`: `export OPENAI_API_KEY="twój_klucz"`

### Pierwszy Projekt

1. **Utwórz Katalog Projektu**:
   ```bash
   # Windows
   mkdir "C:\Users\%USERNAME%\Documents\DocumentAnalysis\Projects\moja_praca"
   
   # macOS
   mkdir -p ~/Documents/DocumentAnalysis/Projects/moja_praca
   ```

2. **Skopiuj Pliki Narzędzia**:
   - Umieść wszystkie 7 plików .py w katalogu projektu
   - Umieść swój dokument (PDF, DOCX, TXT) w tym samym katalogu

3. **Edytuj Konfigurację**:
   - Otwórz `run_document.py`
   - Zmień `PROJECT_NAME = "moja_praca"`
   - Zmień `PROJECT_FILE = "twój_dokument.pdf"`

4. **Uruchom Analizę**:
   ```bash
   python run_document.py
   ```

### Analiza Wyników

Po zakończeniu analizy otrzymasz:

1. **results_doc.csv** - Szczegółowe dane z mapowaniem akapit-klaster
2. **[nazwa_pliku]_analysis.docx** - Raport czytelny dla człowieka
3. **[nazwa_pliku]_analysis.json** - Dane w formacie maszynowym
4. **analysis_report.txt** - Podsumowanie analizy

### Zaawansowana Analiza

1. **Klasyfikacja Treści**:
   ```bash
   python analyze_results_v2.py --project moja_praca
   ```
   
2. **Podsumowania AI**:
   ```bash
   python analyze_results_v3.py
   ```
   
3. **Ekstrakcja Klastrów**:
   ```bash
   python extract_clusters.py
   ```
   
4. **Reprezentatywne Przykłady**:
   ```bash
   python extract_examples.py --project moja_praca --examples 5
   ```

## Interpretacja Statystyk

### Współczynnik Podobieństwa
- **Poniżej 15%**: Dokument ma dobrą różnorodność treści
- **15-30%**: Umiarkowane podobieństwo - przejrzyj największe klastry
- **Powyżej 30%**: Wysokie podobieństwo - rozważ optymalizację treści

### Rozmiary Klastrów
- **2-3 akapity**: Małe powtórzenia, prawdopodobnie naturalne
- **4-10 akapitów**: Średnie klastry wymagające przeglądu
- **Powyżej 10**: Duże klastry - prawdopodobnie znaczące powielanie

## Wskazówki dla Różnych Typów Dokumentów

### Prace Naukowe
- Użyj eps=0.25 dla rygorystycznego wykrywania
- Zwróć uwagę na sekcje metodologii i przeglądu literatury
- Sprawdź powtarzające się definicje i wyjaśnienia

### Raporty Biznesowe
- Użyj eps=0.35 dla szerszego wykrywania tematycznego
- Sprawdź powtarzające się analizy i wnioski
- Zwróć uwagę na duplikujące się opisy wykonawców

### Książki i Artykuły
- Użyj eps=0.4 dla wykrywania tematycznego
- Sprawdź powtarzające się przykłady i ilustracje
- Zwróć uwagę na podobne wprowadzenia do rozdziałów

## Bezpieczeństwo i Prywatność

- **Przetwarzanie Lokalne**: Cała analiza odbywa się na twoim komputerze
- **Przechowywanie Kluczy**: Klucze API przechowywane lokalnie w systemie
- **Brak Wysyłania Danych**: Twoje dokumenty nie opuszczają twojej maszyny (poza opcjonalnymi podsumowaniami AI)

## Wsparcie i Dalsze Kroki

### Dalsze Zasoby
- [Dokumentacja Sentence Transformers](https://www.sbert.net/)
- [Algorytm Klastrowania DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
- [Najlepsze Praktyki OpenAI API](https://platform.openai.com/docs/guides/best-practices)

### Rozwiązywanie Problemów
1. **Sprawdź logi** w katalogu `logs/` twojego projektu
2. **Przetestuj z przykładowym projektem** używając dołączonego przykładowego dokumentu
3. **Uruchom ponownie aplikację** w przypadku błędów
4. **Zainstaluj ponownie** używając instalatora jeśli pliki są uszkodzone

---
*Wersja 1.0 - Dokumentacja w języku polskim dla użytkowników Windows i macOS*