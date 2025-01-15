"""Programma 2"""

import nltk
import sys
import utils
from Corpus import Corpus
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from nltk.corpus import stopwords 
 
# nltk.download('stopwords')



def POSTagging(token):
    """Esegue il POS tagging su una lista di token (parole), 
    assegnando a ciascun token un tag relativo alla sua parte del discorso. Successivamente, 
    calcola la frequenza di ogni coppia (token, tag POS) e raggruppa i token per tag POS in un dizionario.
    Le liste dei token per ciascun tag POS vengono poi ordinate in ordine decrescente di frequenza.
     
    Args:
        token (list): lista di token da taggare

    Returns:
        dict: il dizionario restituito avrà come chiavi i tag POS (ad esempio, 'NN', 'VB', 'JJ') e 
    come valori delle liste di tuple. Ogni tupla contiene il token e la sua frequenza
    associata a quel tag POS. Le liste sono ordinate in modo che i token più frequenti appaiano prima.
    """    
    # Applica il POS tagging ai token utilizzando nltk.tag.pos_tag()
    posList = nltk.tag.pos_tag(token, tagset='universal')
    # Calcola la frequenza di ciascuna coppia (token, tag POS)
    freq = Counter(posList)
    # Dizionario che raggruppa per tag POS
    result_dict = defaultdict(list)
    # Popola il dizionario raggruppato per tag POS
    for (x, y), count in freq.items():
        result_dict[y].append((x, count))
    # Ordina le liste per ogni tag POS in base alla frequenza, in ordine decrescente
    for y in result_dict:
        result_dict[y] = sorted(result_dict[y], key=lambda t: t[1], reverse=True)
    # Restituisce il dizionario ordinato
    return result_dict

def getPOS(posDict, key):
    """
    Restituisce  elementi della lista associata a una chiave in un dizionario.

    Args:
        posDict (dict): Dizionario che associa chiavi a liste di coppie (token, frequenza).
                        Ogni lista è composta da tuple, dove il primo elemento rappresenta il token
                        e il secondo la sua frequenza.
        key (string): La chiave utilizzata per accedere alla lista all'interno del dizionario.

    Returns:
        dict: Un dizionario contenente elementi della lista, trasformati in chiavi
              e valori.

    Raises:
        TypeError: Se il valore associato alla chiave non è una lista o se gli elementi della lista
                   non sono coppie valide per la conversione in dizionario.
        KeyError: Se la chiave non esiste in `posDict`.
    """

    # Recupera la lista di frequenze associata alla chiave 'key' nel dizionario 'posDict'.
    tokenFrequencyList = posDict.get(key)
    
    return tokenFrequencyList

def getTop20Ngrams(lista):
    """conta la frequenza degli ngrammi contenuti nella lista

    Args:
        lista (list): lista di ngrammi

    Returns:
        dict: dizionario dei primi 20 ngrammi ordinato per frequenza discendente 
    """
    return dict(sorted(dict(Counter(lista)).items(), key=lambda x: x[1], reverse=True)[:20])
    
def stopwordsDistribution(token):
    """prende la lista dei token di un corpus, trasforma tutto in minuscole, e calcola la distribuzione delle stopwords rispetto al totale dei token 

    Args:
        token (list): lista dei tokens di un corpus

    Returns:
        float: ditribuzione delle stopwords, arrotondata alla terza cifra decimale
    """    
    tokens = [t.lower() for t in token]
    somma=0 # contatore delle stopwords
    for sw in stopwords.words('english'):
        somma+= tokens.count(sw)
    return round(somma/len(tokens), 3)
    

def main(filePath):
    #inzializzazione dell'istanza per il corpus
    corpus=Corpus(filePath)
    corpus.setText(utils.readFile(filePath))
    corpus.setTokenList(utils.tokenSplitter(corpus.getText()))
    corpus.setSentenceList(utils.sentenceSplitter(corpus.getText()))
    corpus.setVocabulary()
    
    formattedOutput = ''
    
    #Creazione del dizionario che ha come chiave i tag e come valori le liste di tuple contenenti come primo elemento i token associati a quel tag e come secondo elemento la frequenza
    POSDict = dict(POSTagging(corpus.getTokenList()))
    #ADJ-> agg  NOUN->sostantivi  #ADV->AVVERB
    # estraggo i primi 50 aggettivi più frequenti dal dizionario
    top50ADJDict= dict(getPOS(POSDict, "ADJ")[:50])
    # estraggo i primi 50 nomi più frequenti dal dizionario
    top50NOUNDict= dict(getPOS(POSDict, "NOUN")[:50])
    # estraggo i primi 50 avverbi più frequenti dal dizionario
    top50ADVDict= dict(getPOS(POSDict, "ADV")[:50])
    #formatto l'output
    formattedOutput += f"\nTOP 50 AGGETTIVI PRESENTI NEL FILE {corpus.getFileName()}" + utils.createTable(top50ADJDict.values(), ['Frequenza'], top50ADJDict.keys())+"\n"
    formattedOutput += f"\nTOP 50 SOSTANTIVI PRESENTI NEL FILE {corpus.getFileName()}" + utils.createTable(top50NOUNDict.values(), ['Frequenza'], top50NOUNDict.keys())+"\n"
    formattedOutput += f"\nTOP 50 AVVERBI PRESENTI NEL FILE {corpus.getFileName()}" + utils.createTable(top50ADVDict.values(), ['Frequenza'], top50ADVDict.keys())+"\n"
    
    # creo la lista di tutti i monogrammi
    monograms = list(nltk.ngrams(corpus.getTokenList(),1))
    # creo la lista di tutti i bigrammi
    bigrams = list(nltk.ngrams(corpus.getTokenList(),2))
    # creo la lista di tutti i trigrammi
    trigrams = list(nltk.ngrams(corpus.getTokenList(),3))
    # Prendo i primi 20 ordinati per frequenza
    top20Monograms = getTop20Ngrams(monograms)
    top20Bigrams = getTop20Ngrams(bigrams)
    top20Trigrams = getTop20Ngrams(trigrams)
    # formatto loutput
    formattedOutput += f"\nTOP 20 MONOGRAMMI PRESENTI NEL FILE {corpus.getFileName()}:\n"+ utils.createTable(top20Monograms.values(), ['Frequenza'], top20Monograms.keys())+"\n"
    formattedOutput += f"\nTOP 20 BIGRAMMI PRESENTI NEL FILE {corpus.getFileName()}:\n"+ utils.createTable(top20Bigrams.values(), ['Frequenza'], top20Bigrams.keys())+"\n"
    formattedOutput += f"\nTOP 20 MONOGRAMMI PRESENTI NEL FILE {corpus.getFileName()}:\n"+ utils.createTable(top20Trigrams.values(), ['Frequenza'], top20Trigrams.keys())+"\n"
    # Calcolo distribuzione delle stopwords sul testo e formattazione nel file di output
    formattedOutput += f"\n\nLa percentuale di Stopwords presenti nel testo {corpus.getFileName()} e' {stopwordsDistribution(corpus.getTokenList()) * 100}%\n"
    
    # calcolo distribuzione dei pronomi e media per frase #PRON -> pronomi 
    totalePronomi = sum([x for y,x in getPOS(POSDict,"PRON")])
    # formatto l'output inserendo il rapporto tra il numero di pronomi e il totlae dei token ricavato grazie alla lista non ordinata di token contenuta nell'istanza "corpus"
    # Calcolo anche la media di pronomi per frasi grazie alla lista non ordinata di frasi contenuta nell'istanza "corpus"
    formattedOutput +=f"\nNel corpus {corpus.getFileName()} ci sono {totalePronomi}, in rapporto al totale dei token equivale al {round((totalePronomi/len(corpus.getTokenList()))*100,3)}%, per ogni frase ci sono circa {round(totalePronomi/len(corpus.getSentenceList()),3)} pronomi.\n\n"
    print(formattedOutput)
    return

if __name__ == '__main__':
    #safe exit se non è stato passato nessun file
    if len(sys.argv)<2:
        print("Attenzione! Non hai passato nessun file da input.")
    else:
        main(sys.argv[1])
    