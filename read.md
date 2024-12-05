# 🖋️ Tolkning av Handskriven Text med Maskininlärning

Detta projekt syftar till att bygga en maskininlärningsmodell som kan tolka handskriven text (bokstäver och siffror) och konvertera den till digitalt textformat. Verktyget har potential att användas inom flera områden, som automatisering av dokumenthantering, tillgänglighet och historisk forskning.

---

## 📋 Projektmål
1. Utveckla en modell som kan tolka en bred variation av handstilar.
2. Uppnå hög noggrannhet vid översättning av handskriven text till digital text.
3. Minimera vanliga fel, såsom förväxling av liknande tecken (ex. 'O' och '0', eller '1' och 'I').

---

## 📂 Projektstruktur
```plaintext
📁 src
   ├── data_processing.py       # Datahantering och förbearbetning
   ├── model_training.py        # Träning av maskininlärningsmodellen
   ├── model_evaluation.py      # Utvärdering av modellens prestanda
   ├── predict.py               # Prediktion på nya bilder
📁 notebooks
   ├── exploratory_data_analysis.ipynb  # Första datainsikter
   ├── training_logs.ipynb              # Loggar från modellträning
📁 saved_models                  # Sparade modeller
📁 data                         # Dataset och bearbetade data
📄 README.md                    # Projektbeskrivning
📄 requirements.txt             # Nödvändiga Python-paket
```

---

## 📊 Dataset
Dataset som används: EMNIST Letters

   - Innehåll: Handritade bokstäver och siffror.
   - Antal bilder: Cirka 145 000 svartvita bilder i 28x28 pixlar.
   - Fördelar:
      - Stort antal handskrivna tecken för robust träning.
      - Variation av handstilar för generalisering.

# Datapreparation
   - Normalisering: Pixelvärden skalas till intervallet 0-1.
   - Etikettering: Etiketterna representeras med one-hot encoding för modellträning.

---

## 🧠 Modellarkitektur

Använd modell: Convolutional Neural Networks (CNN)

   - CNN är idealiskt för att känna igen mönster i bilder, såsom handskrivna tecken.
   - Startar med en grundläggande arkitektur och utökar vid behov med transfer learning.

# Förbättringar och Utökningar

   - Data Augmentation: Rotationer, skjuvningar och ljusjusteringar för att förbättra robusthet.
   - Transfer Learning: Möjlighet att använda förtränade modeller.
   - Framtida Utveckling: Tolka kompletta ord och sekvenser.

---


## 🛠️ Teknikval

- Språk: Python 3.11
- Bibliotek:
   - TensorFlow och Keras för modellträning.
   - NumPy och Pandas för datahantering.
   - Matplotlib och Seaborn för visualisering.

Installera beroenden med:

```bash
   pip install -r requirements.txt
```

---

## 🚀 Hur du kör projektet

1. Klona repot:
```bash
   git clone https://github.com/username/project-name.git
   cd project-name
```

2. Installera nödvändiga bibliotek:
```bash
   pip install -r requirements.txt
```

3. Ladda ner dataset: Följ instruktionerna i src/data_processing.py för att ladda ner och förbereda EMNIST Letters-datasetet.

4. Träna modellen:
```bash
   python src/model_training.py
```

5. Utvärdera modellen:
```bash
python src/model_evaluation.py
```

6. Gör prediktioner: Lägg till dina bilder i mappen data/new_images och kör:
```bash
   python src/predict.py
```

---

## 📈 Resultat

- Första iterationens noggrannhet: XX.X% (placeholder).
- Confusion Matrix visar att största utmaningen är att skilja på:
   - O och 0
   - 1 och I

# Exempelresultat
| Original Image  | Predicted Text  |
|-----------------|-----------------|
| ![Example1](examples/img1.png) | A |
| ![Example2](examples/img2.png) | 5 |

---

