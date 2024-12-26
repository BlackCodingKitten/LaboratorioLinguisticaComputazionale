"""Programma 1"""

import nltk
import math
import sys
import utils
import re


def writeIncrementalTTR (fileName ,fileTokenList):
    """calcola in maniera incrementale il TTR dei due testi e ne scrive i risultati su una stringa che poi verrà ritornata,
    arriva fino all'ultimo token della lista saltando di 200 token in 200.

    Args:
        fileName (string tuple): tupla che contiene i nomi dei file
        fileTokenList (string list tuple): tupla delle liste di token contenute nei file

    Returns:
       string: ritorna quello che va scritto nel file di output
    """    
    TTR = [0,0] 
    index = 1 #tengo traccia di quante volte salto di 200 token per verificare alla fine se ho incluso tutti i token
    toReturn = f"\n\nTTR INCREMENTALE:\n\nConsideriamo il testo {fileName[0]}:"
    for i in range(200,len(fileTokenList[0]),200):
        toReturn+=f"\n{index}) Nei primi {i} tokens il TTR vale {round(len(set((fileTokenList[0])[0:i]))/i,3)}."
        index+=1
    # Ultima sezione della lista se non viene presa dallo slicing
    if(index*200 <= len(fileTokenList[0])):
        TTR[0]= round(len(set(fileTokenList[0])/len(fileTokenList[0])),3)
        toReturn+=f"\nIl TTR totale del testo {fileName[0]} vale {TTR[0]}."  
    # faccio la stessa cosa per l'altro file
    index = 1 #riazzero il contatore dei salti
    toReturn+=f"\n\nConsideriamo il testo {fileName[1]}:"
    for i in range(200,len(fileTokenList[1]),200):
        toReturn+=f"\n{index}) Nei primi {i} tokens il TTR vale {round(len(set((fileTokenList[1])[0:i]))/i,3)}."
        index+=1
    # Ultima sezione della lista se non viene presa dallo slicing
    if(index*200 <= len(fileTokenList[0])):
        TTR[1]=round((len(set(fileTokenList[1]))/len(fileTokenList[1])),3)
        toReturn+=f"\nIl TTR totale del testo {fileName[1]} vale {TTR[1]}." 
    
    
    toReturn+=f"\nIl TTR maggiore è quello del testo {fileName[TTR.index(max(TTR))]}"
    
    return toReturn+"\n"

def writeSentencesDiff(fileName, fileSentenceList):
    """Calcola la differenza del numero di frasi che contengono i due testi
    
    Args:
        fileName (string tuple): tupla che contiene i nomi dei file
        fileSentenceList (string list tuple): tupla delle liste di frasi contenute nei file

    Returns:
        string: ritorna quello che va scritto nel file di output
    """    
    # calcolo le lunghezze delle liste di frasi di ciascun file 
    len1=len(fileSentenceList[0])
    len2=len(fileSentenceList[1])
    diff=len1-len2
    # se la differenza è negativa la moltiplico per -1 (potevo anche controllare quale dei due fosse maggiore)
    if diff<0:
        diff *= -1
    return f"DIFFERENZA NUMERO DI FRASI:\nIl file {fileName[0]} contiene {len1} frasi.\nIl file {fileName[1]} contiene {len2} frasi.\nLa differenza tra i due corpus risulta {diff} frasi.\n"

def writeTokensDiff(fileName,fileTokenList):
    """Calcola la differenza del numero di parole token che contengono i due testi

    Args:
        fileName (string tuple): tupla che contiene i nomi dei file
        fileTokenList (string list tuple): tupla delle liste di token contenute nei file

    Returns:
       string: ritorna quello che va scritto nel file di output
    """    
    # calcolo le lunghezze delle liste di frasi di ciascun file 
    len1=len(fileTokenList[0])
    len2=len(fileTokenList[1])
    diff=len1-len2
    # se la differenza è negativa la moltiplico per -1 (potevo anche controllare quale dei due fosse maggiore)
    if diff<0:
        diff *= -1
    return f"\nDIFFERENZA NUMERO DI TOKEN:\nIl file {fileName[0]} contiene {len1} token.\nIl file {fileName[1]} contiene {len2} token.\nLa differenza tra i due corpus risulta {diff} token.\n"
 
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
 
def main(filePath1,filePath2):
    
    fileName = (utils.fileName(filePath1),utils.fileName(filePath2))
    # Stringa in cui verrà salvato tutto quello che poi va trascritto sul file
    toWrite = ""
    
    # Leggo il contenuto dei due file passati come argomento e restituisco il testo come unica stringa e lo salvo nella lista fileText
    fileTextList = (utils.readFile(filePath1),utils.readFile(filePath2))
   
    # divido i testi dei sue file in liste di frasi e li salvo in fileSentenceList
    fileSentenceList = (utils.sentenceSplitter(fileTextList[0]),utils.sentenceSplitter(fileTextList[1]))
        
    # divido i file in liste di token e le salvo in fileTokenList
    fileTokenList = (utils.tokenSplitter(fileTextList[0]),utils.tokenSplitter(fileTextList[1]))
    
    # creo la tupla che contiene i Vocabolari dei tipi dei due file 
    fileVocabularyList = (utils.createVocabulary(fileTokenList[0]),utils.createVocabulary(fileTokenList[1]))
    
    print(writeSentencesDiff(fileName,fileSentenceList),writeTokensDiff(fileName,fileTokenList),writeSentencesCharMean(fileName,fileSentenceList),writeTokensCharMean(fileName, fileTokenList), writeIncrementalTTR(fileName,fileTokenList))

main("/home/mikela/Documents/LaboratorioLinguisticaComputazionale/Progetto Finale/data/ChildrenStories_Corpus.txt","/home/mikela/Documents/LaboratorioLinguisticaComputazionale/Progetto Finale/data/Cryptography_Corpus.txt")