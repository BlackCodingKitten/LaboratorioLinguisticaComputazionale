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


def incrementalTTR (tokens):
    """calcola in maniera incrementale il TTR di un testo saltando di 200 token in 200.

    Args:
        tokens (string list): lista dei token
    Returns:
       (str,float) list: ritorna la lista dei valori dei TTR associati alla relativa porzione
    """    
    TTR = []
    index = 1 #tengo traccia di quante volte salto di 200 token per verificare alla fine se ho incluso tutti i token
    for i in range(200,len(tokens),200):
        TTR.append((i,round(len(set(tokens[0:i]))/i,3)))
        index+=1
    # Ultima sezione della lista se non viene presa dallo slicing
    if((len(tokens)%200) != 0):
        TTR.append((len(tokens),round(len(set(tokens))/len(tokens),3)))
    return TTR  
    
def lenListDiff(llist):
    """Calcola la differenza di lunghezza di due liste 

    Args:
       llist (string list tuple): tupla delle liste di token contenute nei file

    Returns:
       int: ritorna il valore assoluto della differenza
    """        
    return abs(len(llist[1])-len(llist[0]))
 
def charMean(lista):
    """calcola la lunghezza media in caratteri delle stringhe nella lista
    
    Args:
        lista (string list): lista delle stringhe su cui calcolare la media 
    Returns:
        float: ritorna la media di caratteri presenti per ciascuna stringa, fino 3 cifre decimali 
    """    
    # Definizione di una regex per rimuovere i segni di punteggiatura, escluso l'apostrofo
    regex = r"[^\w\s']"
    
    # Rimuovo i segni di punteggiatura e calcolo la lunghezza delle frasi in numero di caratteri e calcolo la media sfruttando la funzione sum delle liste
    mean = sum([len(re.sub(regex,'',s)) for s in lista])/len(lista)
    
    return round(mean,3)

def lemmatizer(tokens):
    """calcola la quantità di lemmi distinti in ciascuno dei corpus e ne riporta anche la differenza

    Args:
        fileTokenList (string list tuple): tupla delle liste di token contenute nei file
        
    Returns:
       string list: ritorna la lista di dei lemmi 
    """    
    lemmatzr = WordNetLemmatizer()
    lemma_list = [lemmatzr.lemmatize(t) for t in tokens]
    return sorted(lemma_list)
    

def POSDistribution(fileNameC1,fileNameC2,tokenC1,tokenC2):
    """Calcola la distribuzione di frequenza delle POS per ciascuna delle liste di token e le trasforma in un dataframe Pandas

    Args:
        fileNameC1 (str): Nome del corpus1
        fileNameC2 (str): Nome del corpus2
        tokenC1 (str list): lista token del corpus1
        tokenC2 (str list): lista token del corpus2

    Returns:
        str: dataframe Pandas convertito a stringa che contiene la distribuzione
    """   
    POSDictTuple = {} #Salvo in un unica struttura dati dizionario i POS dei primi 1000 tokens di entrambi i testi
    for token,POS in nltk.tag.pos_tag((tokenC1)[:1000]):
        if POS in POSDictTuple.keys():
            #ho già incontrato il tag, incremento il valore corrispondente al file 1
            POSDictTuple[POS]=[((POSDictTuple[POS])[0])+1,0]
        else:
            #non ho mai incontrato il tag in quastione, aggiungo la cella corrispondente
            POSDictTuple[POS]= [1,0]
            
    for token,POS in nltk.tag.pos_tag((tokenC2)[:1000]):
        if POS in POSDictTuple.keys():
            #ho già il tag salvato nel dizionario, posso averlo già visto nel file precedente o averlo già incontrato in questo, riporto i valori precedenti ed incremento il contatore relativo al secondo file 
            POSDictTuple[POS] =[((POSDictTuple[POS])[0]),((POSDictTuple[POS])[1])+1]
        else:
            #non ho mai incontrato il tag, devo quindi generare una cella nuova
            POSDictTuple[POS]= [0, 1]
    #creo un dataframe in pandas per formattare l'output
    table = [[f"{x/100}%",f"{y/100}%"] for [x,y] in POSDictTuple.values()]
    dataFrame = pd.DataFrame(table, columns=[fileNameC1,fileNameC2], index=POSDictTuple.keys())
    # print(dataFrame)
    return "\n"+dataFrame.to_string(index = True)+"\n"
    
    
    
         
    
def main(filePath1,filePath2):
      
    # definisco le istanze dei due corpus
    c1 = Corpus.Corpus(filePath1)
    c2 = Corpus.Corpus(filePath2)
    
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
    differenzaNumeroToken = lenListDiff((c1.getTokenList(), c2.getTokenList()))
    
    # calcolo la differenza nel numero di frasi
    differenzaNumeroFrasi = lenListDiff((c1.getSentenceList(), c2.getSentenceList()))
    
    #TTR incrementali salvati in un dizionario ed associati al nome del file
    listeTTRincrementali = {c1.getFileName(): incrementalTTR(c1.getTokenList()), c2.getFileName(): incrementalTTR(c2.getTokenList())}
    
    # media dei caratteri per frase salvati in un dizionario ed associati al nome del file
    mediaCharFrasiC1 = charMean(c1.getSentenceList())
    mediaCharFrasiC2 = charMean(c2.getSentenceList())
    mediaCharFrasi = {c1.getFileName(): mediaCharFrasiC1, c2.getFileName(): mediaCharFrasiC2}
    # calcolo la differenza
    differenzaMediaCharFrasi = abs(mediaCharFrasiC2 - mediaCharFrasiC1)
    
     # media dei caratteri per token salvati in un dizionario ed associati al nome del file
    mediaCharTokenC1 = charMean(c1.getTokenList())
    mediaCharTokenC2 = charMean(c2.getTokenList())
    mediaCharToken = {c1.getFileName(): mediaCharTokenC1, c2.getFileName(): mediaCharTokenC2}
    # calcolo della differenza 
    differenzaMediaCharToken = abs(mediaCharTokenC2-mediaCharTokenC1)
    
    #Calcolo numero dei lemmi distinti salvati in ordine alfabetico in un dizionario ed associati al nome del file
    c1.setLemmaList(lemmatizer(c1.getTokenList()))
    c2.setLemmaList(lemmatizer(c2.getTokenList()))
    numeroLemmi = {c1.getFileName(): len(c1.getLemmaList()), c2.getFileName(): len(c2.getLemmaList())}
    
    # calcolo la media dei lemmi per frase arrotondata a 3 cifre decimali
    mediaLemmiPerFrase = {c1.getFileName(): round(len(c1.getLemmaList())/len(c1.getSentenceList()),3), c2.getFileName(): round(len(c2.getLemmaList())/len(c2.getSentenceList()),3)}
    
    # creazione della tabella che rappresenta la distribuzione POS nei primi 1000 tokens
    POSTable = POSDistribution(c1.getFileName(),c2.getFileName(), c1.getTokenList(), c2.getTokenList())
    
   
    
    
    



main("/home/mikela/Documents/LaboratorioLinguisticaComputazionale/Progetto Finale/data/ChildrenStories_Corpus.txt","/home/mikela/Documents/LaboratorioLinguisticaComputazionale/Progetto Finale/data/Cryptography_Corpus.txt")