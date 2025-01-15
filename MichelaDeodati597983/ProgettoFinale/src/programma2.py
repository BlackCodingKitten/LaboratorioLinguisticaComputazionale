"""Programma 2"""

# 1. I top-50 Sostantvi, Avverbi e Aggetvi più frequenti (con rela>va frequenza, ordina> per 
# frequenza decrescente);

import nltk
import sys
import utils
from Corpus import Corpus
import pandas as pd
import numpy as np
from collections import Counter, defaultdict



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
    posList = nltk.tag.pos_tag(token)
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
    return dict(result_dict)


def ngramsListCreator(ngramDim, text):
    ngramLList = []
    for i in range(len(ngramDim)):
        ngramLList.append(list(nltk.ngrams(text,ngramDim[i])))
    return ngramLList    

def main(filePath):
    #inzializzazione dell'istanza per il corpus
    corpus=Corpus(filePath)
    corpus.setText(utils.readFile(filePath))
    corpus.setTokenList(utils.tokenSplitter(corpus.getText()))
    corpus.setSentenceList(utils.sentenceSplitter(corpus.getText()))
    corpus.setVocabulary()
    
    
     
    
    for i in POSTagging(corpus.getTokenList()).items():
        print(i,"\n")
    
    return

if __name__ == '__main__':
    #safe exit se non è stato passato nessun file
    if len(sys.argv)<2:
        print("Attenzione! Non hai passato nessun file da input.")
    else:
        main(sys.argv[1])
    