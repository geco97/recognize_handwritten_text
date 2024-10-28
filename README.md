# Projektbeskrivning: Tolka handskriven text

## Problemdefinition
Målet med detta projekt är att utveckla en maskininlärningsmodell för att tolka handskriven text och konvertera handskrivna tecken till digital text. Detta är användbart inom områden som:
- Automatisering av dokumenthantering
- Tillgänglighet för personer med funktionsnedsättningar
- Förbättrad igenkänning av text i äldre dokument

Genom att tolka handskriven text vill vi skapa ett verktyg som kan tolka bokstäver, siffror och symboler från bilder och översätta dessa till digital text.

## Mål
Projektets huvudsyfte är att:
- Utveckla en modell som kan tolka en bred variation av handstilar
- Översätta handskriven text till ett digitalt format med hög precision
- Undvika vanliga tolkningsfel, t.ex. att särskilja mellan liknande tecken (som 'O' och '0', eller '1' och 'I')

## Dataset
För att träna och testa modellen använder vi ett dataset med bilder av handskrivna bokstäver och siffror, tillsammans med etiketter som anger vilket tecken varje bild representerar.

### Exempel på dataset
1. **MNIST** - Innehåller handskrivna siffror (0-9) i 28x28 pixelbilder. Används ofta för igenkänning av handskrivna siffror.
2. **EMNIST (Extended MNIST)** - Innehåller både bokstäver och siffror och är uppdelat i flera dataset, t.ex. EMNIST-Letters, vilket är användbart för att tolka både bokstäver och siffror.

### Val av dataset
Vi väljer **EMNIST Letters** eftersom det innehåller både handskrivna bokstäver och siffror, vilket matchar projektets mål. Datasetet innehåller cirka 145 000 bilder, vilket ger en robust bas för träning och testning.

### Datakvalitet och format
- **Kvalitet**: EMNIST är ett välanvänt dataset inom maskininlärning med hög kvalitet.
- **Null-värden**: Inga null-värden, då varje bild är kopplad till en etikett.
- **Extrema värden**: Data är normaliserad till 28x28 pixlar, inga extrema värden förekommer.
- **Datatyper**: Bilder representeras som arrayer av pixelvärden (0–255), och etiketten är ett heltal som representerar bokstaven eller siffran.

## Typ av problem
Detta är ett **klassificeringsproblem** där varje bild ska klassificeras till en specifik bokstav eller siffra. Datasetet är labeled, vilket innebär att varje bild har en given etikett som vi kan använda för att träna modellen.

## Val av modell och träningsmetod
Eftersom detta är ett bildklassificeringsproblem är **Convolutional Neural Networks (CNNs)** ett bra val. CNNs har visat sig effektiva på att känna igen mönster i bilder, särskilt vid tolkning av handskriven text. Vi börjar med en enkel CNN-arkitektur och justerar komplexiteten beroende på resultatet.

### Förbättringar
- **Transfer Learning**: Möjligt att använda en förtränad modell på liknande data för ytterligare förbättring.
- **Data Augmentation**: För att öka modellens robusthet genom att introducera små variationer i bilderna, som rotationer eller förändringar i ljusstyrka.

## Databearbetning och förberedelse
För att data ska kunna användas i vår modell genomför vi följande steg:
1. **Normalisering**: Omvandlar pixelvärden från 0-255 till 0-1 för att underlätta modellens träningsprocess.
2. **One-hot encoding**: Konverterar etiketterna till one-hot-kodade format för att göra klasserna distinkta.

## Träningsplattform
Vi kommer att använda **TensorFlow** som träningsplattform tack vare dess bibliotek och kompatibilitet för CNN-modeller.

## Utvärdering av modellens prestanda
Modellens prestanda kommer att utvärderas med hjälp av:
- **Accuracy**: För att mäta hur många bilder modellen klassificerar rätt.
- **Confusion Matrix**: För att analysera vilka bokstäver eller siffror som förväxlas mest och förstå eventuella förbättringsmöjligheter.

--- 
