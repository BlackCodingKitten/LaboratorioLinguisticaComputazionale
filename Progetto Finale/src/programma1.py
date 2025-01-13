"""Programma 1"""

import nltk
from nltk.stem import WordNetLemmatizer
import sys
import utils
import re
import pandas as pd
import Corpus 

# nltk.download("wordnet")
# nltk.download('averaged_perceptron_tagger_eng')


def writeIncrementalTTR (tokens):
    """calcola in maniera incrementale il TTR dei due testi saltando di 200 token in 200.

    Args:
        tokens (string list): lista dei token
    Returns:
       float list: ritorna la lista dei valori dei TTr
    """    
    TTR = []
    index = 1 #tengo traccia di quante volte salto di 200 token per verificare alla fine se ho incluso tutti i token
    for i in range(200,len(list),200):
        TTR.append(round(len(set(list[0:i]))/i,3))
        index+=1
    # Ultima sezione della lista se non viene presa dallo slicing
    if(index*200 <= len(tokens)):
        TTR.append(round(len(set(tokens)/len(tokens)),3))
    return TTR  
    

def lenListDiff(llist):
    """Calcola la differenza di lunghezza di due liste 

    Args:
       llist (string list tuple): tupla delle liste di token contenute nei file

    Returns:
       int: ritorna il valore assoluto della differenza
    """        
    return abs(len(llist[1])-len(llist[0]))
 
def writeSentencesCharMean(fileName, fileSentenceList):
    """calcola la lunghezza media in caratteri delle frsi in enetrmabi i file ne calcola la differenza e restituisce la stringa che va stampata nel file

    Args:
        fileName (string tuple): tupla che contiene i nomi dei file
        fileSentenceList (string list tuple): tupla delle liste di frasi contenute nei file
    Returns:
        string: ritorna quello che va scritto nel file di output
    """    
    # Definizione di una regex per rimuovere i segni di punteggiatura, escluso l'apostrofo
    regex = r"[^\w\s']"
    # Rimuovo i segni di punteggiatura e calcolo la lunghezza delle frasi in numero di caratteri e calcolo la media sfruttando la funzione sum delle liste
    media =[sum(len(re.sub(regex,'',s)) for s in fileSentenceList[0])/len(fileSentenceList[0]), sum(len(re.sub(regex,'',s)) for s in fileSentenceList[1])/len(fileSentenceList[1])]
    diff = round(media[1]-media[0],3) 
    if diff<0:
        diff *= -1
    return f"\n\nMEDIA LUNGHEZZA FRASI:\nLa lunghezza media in caratteri delle frasi nel file {fileName[0]} e' {round(media[0],3)}.\nLa lunghezza media in caratteri della frasi nel file {fileName[1]} e' {round(media[1],3)}.\nLa differenza vale {diff} caratteri."      
    
def writeTokensCharMean(fileName, fileTokenList):
    """calcola la lunghezza media in caratteri delle parole token in enetrmabi i file ne calcola la differenza e restituisce la stringa che va stampata nel file

    Args:
        fileName (string tuple): tupla che contiene i nomi dei file
        fileTokenList (string list tuple): tupla delle liste di token contenute nei file
    Returns:
        string: ritorna quello che va scritto nel file di output
    """  
    # Definizione di una regex per rimuovere i segni di punteggiatura, escluso l'apostrofo
    regex = r"[^\w\s']"
    # Rimuovo i segni di punteggiatura e calcolo la lunghezza dei token in numero di caratteri e calcolo la media sfruttando la funzione sum delle liste
    media =[sum(len(re.sub(regex,'',t)) for t in fileTokenList[0])/len(fileTokenList[0]), sum(len(re.sub(regex,'',s)) for s in fileTokenList[1])/len(fileTokenList[1])]
    diff = round(media[1]-media[0],3) 
    if diff<0:
        diff *= -1
    return f"\n\nMEDIA LUNGHEZZA TOKEN:\nLa lunghezza media in caratteri delle parole token nel file {fileName[0]} e' {round(media[0],3)}.\nLa lunghezza media in caratteri delle parole token nel file {fileName[1]} e' {round(media[1],3)}.\nLa differenza vale {diff} caratteri." 

def writeLemmaCounter(fileName, fileTokenList):
    """calcola la quantità di lemmi distinti in ciascuno dei corpus e ne riporta anche la differenza

    Args:
        fileName (string tuple): tupla che contiene i nomi dei file
        fileTokenList (string list tuple): tupla delle liste di token contenute nei file
        
    Returns:
       string: ritorna quello che va scritto nel file di output
    """    
    lemmatizer = WordNetLemmatizer()
    lemma_list1 = [lemmatizer.lemmatize(t) for t in fileTokenList[0]]
    lemma_list2 = [lemmatizer.lemmatize(t) for t in fileTokenList[1]]
    
    diff = len(set(lemma_list2))-len(set(lemma_list1))
    if diff<0:
        diff *= -1
    return f"\nCOUNTER LEMMI:\nIl file {fileName[0]} contiene {len(lemma_list1)} lemmi.\nIl file {fileName[1]} contiene {len(lemma_list2)} lemmi.\nLa differenza di lemmi tra i due testi equivale a {diff}" 

def writeSentenceMeanLemma(fileName, fileSentenceList):
    """calcola la media dei lemmi distinti per ogni frase dei due corpus 

    Args:
        fileName (string tuple): tupla che contiene i nomi dei file
        fileTokenList (string list tuple): tupla delle liste di token contenute nei file
        
    Returns:
       string: ritorna quello che va scritto nel file di output
    """    
    lemmatizer = WordNetLemmatizer()
    somma = 0
    for s in fileSentenceList[0]: 
        lemmaList = [lemmatizer.lemmatize(t) for t in nltk.tokenize.word_tokenize(s)]
        somma+=len(set(lemmaList))
    toReturn =f"\nMEDIA DEI LEMMI PER FRASE:\nNel file {fileName[0]} per ogni frase ci sono in media {round(s/len(fileSentenceList[0]),3)} lemmi distinti."
    somma=0
    for s in fileSentenceList[1]:
        lemmaList = [lemmatizer.lemmatize(t) for t in nltk.tokenize.word_tokenize(s)]
        somma+=len(lemmaList)
    toReturn+=f"\nNel file {fileName[1]} per ogni frase ci sono in media {round(s/len(fileSentenceList[1]),3)} lemmi distinti."
    return toReturn

def writePOSDistribution(fileName, fileTokenList):
    """per ogni corpus estrae le Part Of Speech dei primi 1000 token e ne calcola la distrubuzione in percentuale
    
    Args:
        fileName (string tuple): tupla che contiene i nomi dei file
        fileTokenList (string list tuple): tupla delle liste di token contenute nei file

    Returns:
       string: ritorna quello che va scritto nel file di output, sottoforma di dataframe pandas
    """    
    POSDictTuple = {} #Salvo in un unica struttura dati dizionario i POS dei primi 1000 tokens di entrambi i testi
    for token,POS in nltk.tag.pos_tag((fileTokenList[0])[:1000]):
        if POS in POSDictTuple.keys():
            #ho già incontrato il tag, incremento il valore corrispondente al file 1
            POSDictTuple[POS]=[((POSDictTuple[POS])[0])+1,0]
        else:
            #non ho mai incontrato il tag in quastione, aggiungo la cella corrispondente
            POSDictTuple[POS]= [1,0]
            
    for token,POS in nltk.tag.pos_tag((fileTokenList[1])[:1000]):
        if POS in POSDictTuple.keys():
            #ho già il tag salvato nel dizionario, posso averlo già visto nel file precedente o averlo già incontrato in questo, riporto i valori precedenti ed incremento il contatore relativo al secondo file 
            POSDictTuple[POS] =[((POSDictTuple[POS])[0]),((POSDictTuple[POS])[1])+1]
        else:
            #non ho mai incontrato il tag, devo quindi generare una cella nuova
            POSDictTuple[POS]= [0, 1]
    #creo un dataframe in pandas per formattare l'output
    table = [[f"{x/100}%",f"{y/100}%"] for [x,y] in POSDictTuple.values()]
    dataFrame = pd.DataFrame(table, columns=[fileName[0],fileName[1]], index=POSDictTuple.keys())
    # print(dataFrame)
    return "\n"+dataFrame.to_string(index = True)+"\n"
    
    
    
         
    
def main(filePath1,filePath2):
      
    # definisco le istanze dei due corpus
    c1 = Corpus(filePath1)
    c2 = Corpus(filePath2)
    
    # Leggo il contenuto dei due file passati come argomento e restituisco il testo come unica stringa
    c1.setText(utils.readFile(filePath1))
    c2.setText(utils.readFile(filePath2))
    
    # divido i testi dei sue file in liste di frasi e li salvo
    c1.setSentenceList(utils.sentenceSplitter(c1.getText()))
    c2.setSentenceList(utils.sentenceSplitter(c2.getText()))
   
    # divido i file in liste di token e le salvo 
    c1.setTokenList(utils.tokenSplitter(c1.getText()))
    c2.setTokenList(utils.tokenSplitter(c2.getText()))
    
    # creo i vocabolari con le frequenze assolute delle parole tipo
    if not (c1.setVocabulary() and  c2.setVocabulary()):
        print ("Errore!! Impossibile creare i vocabolari dei due corpus.")
        
    # calcolo la differenza nel numero di token 
    differenzaNumeroToken = lenListDiff((c1.getTokenList(), c2.getTokeList()))
    # calcolo la differenza nel numero di frasi
    differenzaNumeroFrasi = lenListDiff((c1.getSentenceList(), c2.getSentenceList()))
    
       



main("/home/mikela/Documents/LaboratorioLinguisticaComputazionale/Progetto Finale/data/ChildrenStories_Corpus.txt","/home/mikela/Documents/LaboratorioLinguisticaComputazionale/Progetto Finale/data/Cryptography_Corpus.txt")