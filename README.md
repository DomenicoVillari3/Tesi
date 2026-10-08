# Tesi — Riconoscimento di segni con MediaPipe, LSTM e algoritmo genetico

Progetto sperimentale per il riconoscimento di segni a partire da sequenze video, con esperimenti organizzati per **LIS (Lingua dei Segni Italiana)** e **ASL (American Sign Language)**.

La pipeline estrae landmark o angoli delle mani con **MediaPipe Holistic**, costruisce dataset temporali e addestra reti **LSTM** con TensorFlow/Keras. Un algoritmo genetico esplora il numero e la dimensione degli strati della rete. Uno script di inferenza usa la webcam per mostrare le etichette predette.

Il repository comprende codice, modelli salvati, etichette, conteggi dei frame, grafici e log sperimentali. I video sorgente e gli array di training `x.npy` e `y.npy` non sono inclusi.

## Obiettivi e funzionalità

- Estrarre caratteristiche di mani, posa e, opzionalmente, volto dai video.
- Confrontare coordinate dei landmark e una rappresentazione compatta basata su sei angoli.
- Aumentare i video con trasformazioni spaziali e fotometriche.
- Classificare sequenze temporali mediante reti LSTM.
- Esplorare gli iperparametri della rete con selezione, crossover e mutazione.
- Visualizzare le predizioni in una finestra OpenCV durante l'acquisizione da webcam.

Il sistema classifica le etichette del modello scelto: la concatenazione delle predizioni nell'interfaccia webcam non implementa una traduzione linguistica completa.

## Organizzazione del repository

| File o cartella | Contenuto |
| --- | --- |
| `mp_detection.py` | Acquisizione webcam, estrazione dei landmark e processing dei video |
| `get_angle.py` | Calcolo degli angoli tra punti |
| `coordinate_sferiche.py` | Conversione cartesiana → sferica; l'uso nell'estrazione è commentato |
| `augment.py` | Funzioni di trasformazione dei video |
| `preprocessing.py` | Pipeline di augmentation e scrittura delle etichette |
| `create_dataset.py` | Lettura delle sequenze, padding e generazione di `x.npy` e `y.npy` |
| `model.py` | Costruzione, training e valutazione della rete LSTM |
| `inference.py` | Classificazione delle sequenze acquisite dalla webcam |
| `NeuralNetwork.py` | Rete parametrica usata nella ricerca genetica |
| `Individual.py` | Individuo, iperparametri, training e fitness |
| `Population.py` | Popolazione, selezione, crossover e mutazione |
| `Algorithm.py` | Ciclo evolutivo e registrazione dei risultati |
| `LIS_10/` | Modelli, etichette e conteggi dei frame degli esperimenti LIS |
| `plain_ASL/` | Artefatti degli esperimenti ASL |
| `argumented_ASL/` | Artefatti degli esperimenti ASL con augmentation |
| `genetico/` | Log CSV e individui degli esperimenti genetici |
| `fig_def/` | Grafici di accuracy, loss e matrici di confusione |
| `weights.keras` | Modello salvato nella radice |
| `labels.txt` / `frames.txt` | Etichette ordinate e conteggi dei frame nella radice |

Gli script usano prevalentemente percorsi relativi alla directory corrente. Esegui i comandi dalla radice del progetto e conserva insieme modello, etichette e dati dell'esperimento.

## Requisiti e ambiente

Le dipendenze ricavate dagli import sono:

| Libreria | Uso |
| --- | --- |
| TensorFlow | Backend e utility per la preparazione dei dati |
| Keras | Reti LSTM, layer e callback |
| MediaPipe | Rilevamento Holistic |
| OpenCV (`cv2`) | Lettura/scrittura video, webcam e finestre |
| NumPy | Array numerici e file `.npy` |
| scikit-learn | Split del dataset e metriche |
| imbalanced-learn | SMOTE |
| Matplotlib | Grafici e matrici di confusione |

Per la webcam è necessario un ambiente desktop con accesso alla videocamera e supporto alle finestre OpenCV.

### Preparazione

```bash
git clone https://github.com/DomenicoVillari3/Tesi.git
cd Tesi
python -m venv .venv
```

Attiva l'ambiente su Linux/macOS:

```bash
source .venv/bin/activate
```

Su Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Installa versioni compatibili delle librerie elencate. Il repository non fornisce `requirements.txt`, un lockfile o la versione dell'ambiente originale: l'installazione va quindi verificata prima di eseguire gli esperimenti.

### Compatibilità MediaPipe

Il codice usa l'API legacy `mp.solutions.holistic`. Le versioni recenti di MediaPipe che non espongono `mp.solutions` non possono eseguire gli script senza adattamenti.

Per riprodurre il percorso legacy, **MediaPipe 0.10.21** è un candidato da verificare insieme alle versioni di Python, TensorFlow, Keras, NumPy e OpenCV. Non costituisce un ambiente completo già validato per questo repository.

Riferimenti: [discussione ufficiale sulla rimozione delle Solutions](https://github.com/google-ai-edge/mediapipe/issues/6204) e [compatibilità del tutorial Holistic](https://github.com/google-ai-edge/mediapipe/issues/6241).

Controlla gli import e l'API richiesta:

```bash
python -c "import tensorflow, keras, cv2, numpy, sklearn, imblearn, matplotlib; import mediapipe as mp; print(mp.__version__); print(mp.solutions.holistic.Holistic)"
```

Dopo aver validato l'ambiente, registra le dipendenze effettive:

```bash
python -m pip freeze > requirements-local.txt
```

## Rappresentazioni delle caratteristiche

`mp_detection.py` implementa tre modalità:

| Modalità | Caratteristiche per frame | Comando |
| --- | ---: | --- |
| Mani + posa | 171: 21 landmark per mano e 15 di posa, con coordinate x/y/z | `python mp_detection.py --process` |
| Mani + posa + volto | 1575: le precedenti più 468 landmark facciali x/y/z | `python mp_detection.py --process --face` |
| Angoli delle mani | 6: tre angoli per mano | `python mp_detection.py --process --angles` |

L'ordine delle coordinate è **mano sinistra, mano destra, posa, volto opzionale**. La rappresentazione angolare concatena gli angoli della mano sinistra e della mano destra.

Durante l'estrazione dai video, i frame senza rilevamenti delle mani utilizzabili sono scartati; le parti mancanti di un rilevamento sono rappresentate con zeri secondo le funzioni di estrazione.

## Workflow: dai video al modello

### 1. Prepara i video e le etichette

Crea una directory locale `video/` e inserisci video MP4 con nomi nel formato:

```text
ciao_0.mp4
ciao_1.mp4
grazie_0.mp4
grazie_1.mp4
```

La parte prima di `_` identifica l'etichetta. `labels.txt` deve contenere una classe per riga, nell'ordine usato per gli indici di output della rete.

Per un nuovo dataset usa nomi coerenti e preferibilmente senza spazi o caratteri speciali. Non cambiare l'ordine o il significato delle etichette associate a un modello già addestrato.

In `mp_detection.py`, sostituisci il percorso assoluto presente in `VIDEO_DIR` con la directory dei tuoi video, ad esempio:

```python
VIDEO_DIR = "video"
```

I percorsi delle sequenze e del conteggio dei frame sono controllati da `POINTS_DIR_NAME` e `FRAMES_FILENAME`, rispettivamente `points` e `frames.txt` per default.

### 2. Augmentation opzionale

`augment.py` comprende flip, rumore, variazioni di luminosità/contrasto, blur, resize, color jitter, rotazioni, traslazioni e cambiamento degli FPS.

Per applicare la pipeline di `preprocessing.py`:

```bash
python preprocessing.py
```

Lo script legge `video/`, scrive i video trasformati nella stessa directory e riscrive `labels.txt`. Eseguilo su una copia dei video sorgente e conserva le etichette originali.

La versione corrente usa un `set` per scrivere le etichette e presenta collisioni tra alcuni nomi di output. Prima di usarla in un esperimento riproducibile, rendi deterministico l'ordine delle classi e assegna nomi univoci ai video generati. Ripetere l'augmentation sulla stessa cartella può elaborare anche file già aumentati.

### 3. Estrai le caratteristiche

Scegli **una sola rappresentazione** per ciascun dataset. Esempio con 171 caratteristiche:

```bash
python mp_detection.py --process
```

Alternative:

```bash
python mp_detection.py --process --angles
python mp_detection.py --process --face
```

Ogni video genera `points/<etichetta>_<numero>.npy`; lo script scrive anche `frames.txt`.

Le modalità condividono la directory di output: separa i risultati se vuoi confrontarle, evitando di sovrascrivere o mescolare sequenze di dimensioni diverse.

Per controllare visivamente i rilevamenti dalla webcam:

```bash
python mp_detection.py --camera
```

Premi **q** per chiudere la finestra.

### 4. Crea il dataset

Con le sequenze in `points/` e le etichette in `labels.txt`:

```bash
python create_dataset.py --create
```

Lo script genera:

| File | Forma | Contenuto |
| --- | --- | --- |
| `x.npy` | `(N, T, F)` | N sequenze, T timestep, F caratteristiche |
| `y.npy` | `(N, C)` | Etichette one-hot per C classi |

Le sequenze vengono uniformate alla lunghezza massima con **padding finale a -99**. Il modello usa `Masking(mask_value=-99)`.

Il matching dei file usa l'etichetta all'interno di una regex. Nel codice corrente i caratteri speciali non sono escapati: ad esempio `come stai?` nella radice può non corrispondere al nome del file atteso. Per supportare questi nomi occorre usare `re.escape(filtro)` nella regex, mantenendo la corrispondenza con le etichette del modello.

### 5. Addestra la rete LSTM

```bash
python model.py --train
```

L'architettura combina:

- Masking dei timestep di padding.
- Uno o più strati LSTM.
- Batch Normalization e Dropout pari a 0,3.
- Strati Dense con attivazione ReLU.
- Un output softmax con una unità per classe.

Gli iperparametri selezionati da `model.py` dipendono dal numero di caratteristiche:

| F | Strati LSTM | Unità LSTM | Strati Dense | Unità Dense |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 4 | 78 | 2 | 49 |
| 171 | 1 | 68 | 1 | 85 |
| 1575 | 1 | 190 | 1 | 99 |
| Altro | 1 | 72 | 1 | 190 |

Il training usa Adam e categorical cross-entropy, fino a 2000 epoche, con EarlyStopping sulla validation loss, patience 5 e ripristino dei pesi migliori.

Se trova `weights.keras`, tenta di caricarlo prima del training. Il checkpoint salva poi il miglior modello nello stesso percorso: conserva una copia degli artefatti che vuoi mantenere e verifica la compatibilità della rete prima del caricamento.

Alla fine vengono mostrati loss, accuracy, curve di training/validation e matrice di confusione.

## Inferenza con webcam

Prima di avviare l'inferenza devono essere disponibili nella directory corrente:

- `weights.keras`, compatibile con l'architettura scelta.
- `labels.txt`, con lo stesso ordine delle classi usato in training.
- `x.npy`, per ricavare lunghezza della sequenza e numero di caratteristiche.

Per il percorso standard a 171 caratteristiche:

```bash
python inference.py
```

Per il percorso a sei angoli, con dati e checkpoint corrispondenti:

```bash
python inference.py --angles
```

Lo script apre la webcam con indice **0**, accumula i frame e applica una soglia di confidenza di **0,65**. La finestra visualizza fino alle ultime cinque etichette accettate, evitando ripetizioni consecutive. Dopo una predizione accettata la sequenza viene azzerata.

Premi **q** per terminare.

Il fallback interattivo quando `x.npy` è assente non è funzionante: il codice continua a usare la variabile `x` non inizializzata e non converte gli input in interi. Va corretto prima di eseguire inferenza senza il dataset.

Per il modello a **1575 caratteristiche**, la gestione dei flag dell'estrattore deve essere adattata anche nell'inferenza: il percorso webcam corrente non attiva automaticamente l'estrazione del volto in base alla forma dell'input.

## Ricerca con algoritmo genetico

L'algoritmo cerca quattro iperparametri:

| Gene | Intervallo |
| --- | --- |
| Numero di strati LSTM | 1–4 |
| Unità degli strati LSTM | 32–256 |
| Numero di strati Dense | 1–5 |
| Unità degli strati Dense | 32–256 |

La configurazione corrente usa **20 individui**, crossover a un punto e probabilità di mutazione del **10%**. Ogni rete usa fino a 50 epoche, con EarlyStopping di patience 10. Il tempo di training viene registrato; la fitness effettiva coincide con l'accuracy di valutazione.

Il ciclo si arresta quando il miglior individuo raggiunge fitness **≥ 0,85**. Questa è una soglia nel codice, non un risultato garantito; non è configurato un limite massimo di generazioni.

### Preparazione necessaria

1. Fornisci `x.npy`, `y.npy` e `labels.txt` coerenti.
2. Correggi `NeuralNetwork.load_data()`: applica `to_categorical` solo se le etichette sono indici interi. `create_dataset.py` produce già una matrice one-hot, che il codice genetico attuale riconverte erroneamente.
3. In `Algorithm.py`, scegli la sorgente della popolazione. Il default legge `individui.txt` nella radice, che non è incluso. Per creare una popolazione nuova usa `Population(size=20, file=None)`; per riprendere un esperimento seleziona un log compatibile e configura `resume=True` e il numero della generazione.
4. Gestisci il caso di fitness tutte uguali in `normalize_fitness()`, che altrimenti divide per zero.

Dopo queste modifiche:

```bash
python Algorithm.py
```

Il ciclo scrive `Generazione.csv` e `individui.txt`. I campi CSV sono `generazione`, `dna`, `tempo`, `accuracy`, `fitness` e `normalized_fitness`.

La ricerca registra gli iperparametri e le metriche, ma non salva automaticamente il modello del miglior individuo. Per usarlo nell'inferenza occorre costruire, addestrare e salvare la rete corrispondente.

I log degli individui vengono letti con `eval()`: usa soltanto file fidati; per un formato di persistenza robusto sostituisci questa lettura con una serializzazione strutturata.

## Artefatti degli esperimenti

| Percorso | Artefatti presenti |
| --- | --- |
| `LIS_10/multi/angles6/` | `weights.keras`, `labels.txt`, `frames.txt` |
| `LIS_10/multi/points171/` | `weights.keras`, `labels.txt`, `frames.txt` |
| `LIS_10/multi/points1575/` | `weights.keras`, `labels.txt`, `frames.txt` |
| `LIS_10/single/` | `weights.keras`, `labels.txt`, `frames.txt` |
| `plain_ASL/angles23/` | `weights.keras`, `labels.txt`, `frames.txt` |
| `plain_ASL/points15/` | `weights.keras`, `labels.txt`, `frames.txt` |
| `argumented_ASL/points_69/` | `weights.keras`, `labels.txt`, `frames.txt` |
| `genetico/6/`, `genetico/171/`, `genetico/575/` | `Generazione.csv` e `individui.txt` |
| `fig_def/` | Grafici PNG di accuracy, loss e matrici di confusione |

I nomi delle cartelle identificano gli esperimenti; non sostituiscono i metadata del modello. Prima di riutilizzare un checkpoint verifica forma di input, architettura, preprocessing e ordine delle classi.

Le etichette nella radice sono, in ordine: **buonanotte, grazie, libro, cane, corpo, acqua, come stai?, ciao, bacio, io, buongiorno**.

Gli esperimenti LIS multi usano **11 etichette**, nonostante il nome `LIS_10`, e riportano `come-stai` al posto di `come stai?`. Gli esperimenti ASL a punti letti dal repository hanno 15 etichette, ma l'ordine differisce tra `plain_ASL` e `argumented_ASL`: mantieni sempre il file associato al checkpoint.

## Valutazione e riproducibilità

I grafici e i log sono artefatti sperimentali; il repository non fornisce i video o il protocollo completo per riprodurre e verificare le metriche.

Nel training corrente di `model.py`:

- SMOTE viene applicato **prima** dello split dei dati.
- Vengono creati train, validation e test, ma `model.fit()` usa `x_test, y_test` come validation data.
- Gli stessi dati vengono poi usati nella valutazione finale.

Per ottenere una stima indipendente delle prestazioni, separa i dati prima dell'oversampling, applicalo solo al training e usa validation e test distinti. Mantieni nello stesso split i video derivati dallo stesso originale e definisci una separazione per partecipante quando il dataset lo consente.

La ricerca genetica valuta gli individui sullo split che guida la selezione: conserva inoltre un test finale indipendente dalla ricerca.

Registra versione delle librerie, seed, mapping delle etichette, rappresentazione delle caratteristiche, split e checkpoint di ogni esperimento.

## Problemi comuni

| Problema | Causa o verifica |
| --- | --- |
| `mediapipe` non espone `solutions` | Versione incompatibile con l'API legacy |
| Directory video non trovata | Aggiorna `VIDEO_DIR` in `mp_detection.py` |
| `x.npy` o `y.npy` mancanti | Estrai le sequenze e avvia `create_dataset.py --create` |
| Classe senza sequenze | Controlla nomi dei file, etichette e caratteri speciali nella regex |
| Shape incompatibile | Non mescolare angoli, coordinate e modelli di esperimenti diversi |
| Pesi incompatibili | Verifica numero di classi e parametri della rete |
| Errore nella ricerca genetica | Controlla la ricodifica one-hot e il percorso di `individui.txt` |
| Webcam non disponibile | Controlla indice della videocamera e permessi del sistema |
| Finestra OpenCV non disponibile | Usa un ambiente desktop e un pacchetto OpenCV con GUI |




## Autore

[Domenico Villari](https://github.com/DomenicoVillari3)

Repository: [DomenicoVillari3/Tesi](https://github.com/DomenicoVillari3/Tesi)

