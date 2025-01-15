# Note 

## Programma 1

### Path dei file:
I percorsi inseriti all'interno del programma sono percorsi assoluti ricavati tramite la libreria "os" e il comando "getcwd()", per una corretta esecuzione senza intoppi usare come cartella di lavoro **ProgettoFinale**, tutto il codice è contenuto nella cartella src, i due programmi con il main si chiamano "programma1.py" e "programma2.py"

### File di Output: 
I file di output sono genrati automaticamente all'interno della cartella results, ciascun file è formattato il più possibile per grantire una lettura efficacie delle informazioni richieste

### Sentiment-classifier: 
Per quanto riguarda il classificatore di polarità del documento, è stato addestrato e salvato tramite la libreria pickle, quindi durante l'esecuzione viene ricaricato quello esistente, ho lasciato il codice di addestramento (funzione **polarityClassificationTrainer()** in programma1.py), ma non viene mai utilizzato durante l'esecuzione. 

### Training-Test set:
Preso da questo link : [https://ai.stanford.edu/~amaas/data/sentiment/](https://ai.stanford.edu/~amaas/data/sentiment/)
## Programma2

## Altro
### ProgettoFinale/src/utils.py e ProgettoFinale/src/Corpus.py
Si tratta di due piccoli file di supporto per non dover scrivere sempre le stesse funzioni tra i due programmi.