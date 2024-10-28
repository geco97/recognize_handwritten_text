# Projekt: Tolkning av handskriven text med hjälp av maskininlärning

## 1. Projektbeskrivning
Detta projekt syftar till att bygga en maskininlärningsmodell som kan tolka handskriven text och konvertera den till digitalt textformat. Målet är att skapa ett verktyg som kan tolka både bokstäver och siffror från bilder och översätta dessa till digital text med hög noggrannhet. Detta är till stor nytta inom flera områden, som:
- **Automatisering av dokumenthantering**: För snabbare insamling och analys av handskrivna dokument.
- **Tillgänglighet**: Hjälpa personer med syn- och motoriska funktionsnedsättningar att använda handskrivna anteckningar digitalt.
- **Historisk forskning**: Tolkning av äldre, handskrivna dokument där digitala kopior inte existerar.

## 2. Projektmål
Det primära målet med projektet är att:
- Utveckla en modell som kan tolka en bred variation av handstilar.
- Uppnå hög noggrannhet vid översättning av handskriven text till ett digitalt format.
- Minimera vanliga fel, såsom förväxling av liknande tecken (ex. 'O' och '0', eller '1' och 'I').

## 3. Val av dataset
För att träna och testa modellen behövs ett dataset som innehåller handskrivna tecken med motsvarande etiketter för varje bild.

### Alternativ för dataset
1. **MNIST (Modified National Institute of Standards and Technology)**:
   - Innehåller handskrivna siffror (0-9) i 28x28 pixelbilder.
   - Är ett vanligt dataset för träning och test av bildigenkänning, särskilt på siffror.

2. **EMNIST (Extended MNIST)**:
   - Innehåller både bokstäver och siffror samt en större variation av handstilar än MNIST.
   - Delar upp i flera deldataset, inklusive EMNIST-Letters som inkluderar både siffror och bokstäver.
   - Valet föll på **EMNIST Letters** för detta projekt, då det matchar projektets mål att tolka både bokstäver och siffror och innehåller cirka 145 000 bilder, vilket ger en robust bas för träning och testning.

### Dataformat och kvalitet
- **Bildformat**: Svartvita bilder med en upplösning på 28x28 pixlar.
- **Kvalitet och noggrannhet**: EMNIST är väl använt inom maskininlärning och har hög kvalitet utan null-värden.
- **Etiketter**: Varje bild har en given etikett som representerar tecknet, vilket underlättar klassificeringen.

## 4. Problemtyp och modeller
Detta är ett **klassificeringsproblem**, där varje bild ska klassificeras till en specifik bokstav eller siffra. Data är etiketterad, vilket gör övervakad maskininlärning lämplig.

### Val av modell
För bildklassificering används ofta **Convolutional Neural Networks (CNN)**, vilket gör dem till ett naturligt val även i detta projekt. CNN är beprövade för att känna igen mönster i bilder och är särskilt användbara för handskriven textigenkänning.

#### Möjliga alternativ och vidareutveckling
- **Enkel CNN-arkitektur**: Starta med en basmodell för att hantera mindre mängder data och få en snabb inblick i resultat.
- **Transfer Learning**: Vid behov av ökad noggrannhet kan en förtränad modell (på liknande data) användas.
- **Data Augmentation**: För att öka modellens robusthet mot variationer i handstilar, genom att introducera små variationer som rotationer och ljusförändringar.

### Utvärdering av prestanda
Prestandan kommer att utvärderas genom:
- **Accuracy**: Mäta hur många bilder modellen klassificerar korrekt.
- **Confusion Matrix**: För att se vilka tecken som förväxlas och analysera möjligheter till förbättringar.

## 5. Databearbetning och förberedelse
För att förbereda data för modellen genomförs följande steg:
1. **Normalisering**: Pixelvärden omvandlas från 0-255 till 0-1, vilket underlättar modellens träningsprocess.
2. **One-hot encoding**: Konverterar etiketterna till en one-hot-kodning, vilket gör det möjligt för modellen att förstå klasserna som distinkta utgångar.

## 6. Träningsplattform och teknikval
Vi kommer att använda **TensorFlow** och **Keras** för att bygga och träna modellen. TensorFlow är väl lämpat för bildklassificering och erbjuder stöd för CNNs och verktyg för dataaugmentation, vilket gör det till en idealisk plattform för detta projekt.

## 7. Framtida utvecklingsmöjligheter
Vid positivt resultat och vidare intresse kan projektet utökas för att hantera mer avancerade funktioner, såsom:
- Tolka kompletta ord istället för enskilda bokstäver eller siffror.
- Utveckla stöd för multipla språk och symboler.
- Förbättra modellen genom användning av mer avancerade djupinlärningstekniker, såsom RNNs för sekvenstolkning av text.

## 8. Lärandemål
Genom detta projekt vill jag lära mig att bygga och träna en CNN-modell för bildigenkänning. Jag är särskilt intresserad av att förstå:
- Hur man hanterar och bearbetar handskrivna tecken
- Djupare kunskaper om CNNs och deras arkitektur
- Hur olika träningsstrategier och dataaugmentation påverkar prestanda och noggrannhet

Med detta projekt vill jag bygga en modell som effektivt kan tolka handskrivna tecken och som kan skalas upp för att bli ett användbart verktyg inom textigenkänning.

---