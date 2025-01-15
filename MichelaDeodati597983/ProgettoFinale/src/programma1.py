"""Programma 1"""

import sys
import nltk
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
import numpy as np
import Corpus
import utils

# nltk.download("wordnet")
# nltk.download('averaged_perceptron_tagger_eng')
nltk.download('universal_tagset')

# tools per la sentiment analisys
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# per salvare il modello 
import pickle

# per ignorare il warning generato dall'assegnare un nuovo tokenizer al TFidfVectorizer -> "/home/mikela/.local/lib/python3.8/site-packages/sklearn/feature_extraction/text.py:525: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'"
import warnings
warnings.filterwarnings(
    'ignore', 
    message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None"
)

def polarityClassificationTrainer():
    """Funzione che crea, addestra e salva un modello di sentiment analysis.
    Il modello viene creato utilizzando un pipeline composta da un vettorizzatore TF-IDF
    e un classificatore Naive Bayes multinomiale. Dopo l'addestramento, il modello viene 
    salvato su file per poter essere riutilizzato senza doverlo ricreare da zero.
    """
    # Definisco il percorso della directory contenente il dataset di training    
    trainingDir = "./sentiment-classifier/sentiment_training"
     # Carico il dataset di training dalla directory, con shuffle per mescolare i dati
    trainingDataSet = load_files(trainingDir, shuffle=True)
    # Divido il dataset in training set (80%) e test set (20%)
    xTrain, xTest, yTrain, yTest = train_test_split(trainingDataSet.data, trainingDataSet.target,test_size = 0.20, random_state = 24)
    
    # Inizializzo un vettorizzatore TF-IDF 
    vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
     # Inizializzo un classificatore Naive Bayes multinomiale
    classifier = MultinomialNB()
     # Creo una pipeline che combina il vettorizzatore TF-IDF con il classificatore Naive Bayes
    pipeline = Pipeline([
        ('tfidf-vectorizer', vectorizer),
        ('multinomialNB', classifier)
    ])
    # Addestro il modello utilizzando il training set
    trained_model = pipeline.fit(xTrain,yTrain)
    
    # Utilizzo il modello addestrato per fare previsioni sul test set
    predicted = trained_model.predict(xTest)
    # Stampo le previsioni e il report di classificazione
    print(predicted)
    print(classification_report(yTest, predicted, target_names = trainingDataSet.target_names))
    
    # Salvo il modello addestrato in un file .pkl per un uso futuro
    fileptr = open("./sentiment-classifier/sentiment-classifier.pkl", "wb")
    pickle.dump(pipeline, fileptr)
    return
    
def polaritySentenceClassification(sentenceList):
    """
    Funzione che analizza una lista di frasi per determinare la distribuzione delle polarità 
    (positiva e negativa) utilizzando un modello di sentiment analysis precedentemente addestrato.
    
    Args:
        sentenceList (list): Lista di frasi da analizzare.

    Returns:
        dict: Dizionario contenente la distribuzione percentuale di sentiment negativo e positivo.
    """
     # Carico il modello di sentiment analysis precedentemente salvato
    sentimentPipeline = pickle.load(open("./sentiment-classifier/sentiment-classifier.pkl", "rb"))
     # Utilizzo il modello per prevedere le polarità delle frasi nella lista
    predictions = sentimentPipeline.predict(sentenceList)
    # Inizializzo i contatori per frasi negative e positive
    neg = 0     #contatore per i negativi
    pos = 0     #contatore per i positivi
    # Itero sulle predizioni per calcolare il numero di frasi positive e negative
    for predictionValue in predictions:
        if predictionValue==1: # Se la previsione è 1, la frase è positiva
            pos+=1
        else:  # Altrimenti, la frase è negativa
            neg+=1
    # Calcolo le distribuzioni percentuali di sentiment negativo e positivo
    return {"Negative Distribution %": round(100*(neg/len(sentenceList)),2), "Positive Distribution %": round(100*(pos/len(sentenceList)),2)}

def polarityCorpusClassification(sentenceList):
    """
    Funzione che analizza un corpus di frasi per determinare la polarità complessiva
    (positiva o negativa) basandosi sulle previsioni di un modello di sentiment analysis.
    
    Args:
        sentenceList (list): Lista di frasi da analizzare.

    Returns:
        str: Una stringa che indica la polarità complessiva del corpus:
             "Positiva" o "Negativa".
    """
    sentimentPipeline = pickle.load(open("./sentiment-classifier/sentiment-classifier.pkl", "rb"))
    predictions = sentimentPipeline.predict(sentenceList)
    # print(predictions)
    neg = 0
    pos = 0
    for v in predictions:
        if v==1:
            pos+=1
        else:
            neg-=1
    # Determino la polarità complessiva in base al bilancio calcolato
    if neg+pos >0:
        return "Positiva"
    elif neg+pos<0:
        return "Negativa"
    else:
        return "Neutro"
    
def incrementalTTR (tokens):
    """calcola in maniera incrementale il TTR di un testo saltando di 200 token in 200.

    Args:
        tokens (string list): lista dei token
    Returns:
       (str,float) list: ritorna la lista dei valori dei TTR associati alla relativa porzione
    """    
    TTR = []
    for i in range(200, len(tokens) + 1, 200):  # Include anche l'ultimo gruppo completo
        TTR.append(round(len(set(tokens[:i])) / i, 3))
    # Controlla se rimangono token non inclusi in un blocco di 200
    if len(tokens) % 200 != 0:
        TTR.append(round(len(set(tokens)) / len(tokens), 3))
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
    lemma_list = [lemmatzr.lemmatize(t.lower()) for t in tokens]
    return set(sorted(lemma_list))
    
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
    POSDictTuple = {} #Salvo in un unica struttura dati (dizionario) i POS dei primi 1000 tokens di entrambi i testi
    for token,POS in nltk.tag.pos_tag((tokenC1)[:1000], tagset='universal'):
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
    return createTable([[f"{x/10}%",f"{y/10}%"] for [x,y] in POSDictTuple.values()], [fileNameC1,fileNameC2], POSDictTuple.keys())#i valori sono calcolati in percentuale (x/1000)*100-> x/10 -> valore%

def createTable(tValue, dfColumns, dfIndex, indexFlag = True):
    """funzione che crea un dataframe pandas con i valori passati e li formatta in una stringa

    Args:
        tValue (list): lista dei valori da inserire nella tabella 
        dfColumns (list): nome delle colonne da inserire
        dfIndex (list): indicizzazione per rirga della tabella
    """    
     #creo un dataframe in pandas per formattare l'output
    dataFrame = pd.DataFrame(tValue, columns=dfColumns, index=dfIndex)
    
    return "\n"+dataFrame.to_string(index = indexFlag)+"\n"

    
def main(filePath1,filePath2, resultfilePath = utils.crateResultsFilePath()):
    
    # definisco le istanze dei due corpus
    c1 = Corpus.Corpus(filePath1)
    c2 = Corpus.Corpus(filePath2)
    
    # definisco e inizializzo la variabile in cuoi salverò l'output formattatato
    formattedOutput = ""
    
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
        
    # calcolo la differenza nel numero di token e formatto l'output
    differenzaNumeroToken = lenListDiff((c1.getTokenList(), c2.getTokenList()))
    # scrivo i risultati formattandoli 
    formattedOutput += createTable([[len(c1.getTokenList()),len(c2.getTokenList()),differenzaNumeroToken]], [c1.getFileName(),c2.getFileName(),"Differenza"],["Numero di Token"])+"\n\n"
    
    # calcolo la differenza nel numero di parole tipo 
    differenzaNumeroType = lenListDiff((c1.getVocabulary().keys(), c2.getVocabulary().keys()))
    formattedOutput += createTable([[len(c1.getVocabulary().keys()),len(c2.getVocabulary().keys()),differenzaNumeroType]], [c1.getFileName(),c2.getFileName(),"Differenza"],["Numero Parole Tipo"])+"\n\n"
    
    # calcolo la differenza nel numero di frasi
    differenzaNumeroFrasi = lenListDiff((c1.getSentenceList(), c2.getSentenceList()))
    formattedOutput += createTable([[len(c1.getSentenceList()),len(c2.getSentenceList()),differenzaNumeroFrasi]], [c1.getFileName(),c2.getFileName(),"Differenza"],["Numero di Frasi"])+"\n\n"
    
    #TTR incrementali salvati in un dizionario ed associati al nome del file econtrollo se le liste hanno la stessa diemensione e in caso contrario riempo i vuoti
    incrementalTTRc1 = incrementalTTR(c1.getTokenList())
    incrementalTTRc2 = incrementalTTR(c2.getTokenList())  
    
    maxLen = max(len(incrementalTTRc1),len(incrementalTTRc2))
    incrementalTTRc1 += [np.nan]*(maxLen-len(incrementalTTRc1))
    incrementalTTRc2 += [np.nan]*(maxLen-len(incrementalTTRc2))
    # fomratto l'output
    index = [f"{i*200} token" for i in range(1,maxLen)]
    index.append(f"{max(len(c1.getTokenList()),len(c2.getTokenList()))} token")
    df = pd.DataFrame({c1.getFileName():incrementalTTRc1,c2.getFileName():incrementalTTRc2}, index)
    formattedOutput += "TTR INCREMENTALE A CONFRONTO:\n\n"+df.to_string()+"\n\n"
    
    # media dei caratteri per frase salvati in un dizionario ed associati al nome del file
    mediaCharFrasiC1 = charMean(c1.getSentenceList())
    mediaCharFrasiC2 = charMean(c2.getSentenceList())
   
    # calcolo la differenza
    differenzaMediaCharFrasi = abs(mediaCharFrasiC2 - mediaCharFrasiC1)
    formattedOutput += createTable([[mediaCharFrasiC1,mediaCharFrasiC2,differenzaMediaCharFrasi]],[c1.getFileName(),c2.getFileName(),"Differenza"],["Media N caratteri per frase"])+"\n\n"
    
     # media dei caratteri per token 
    mediaCharTokenC1 = charMean(c1.getTokenList())
    mediaCharTokenC2 = charMean(c2.getTokenList())
    # calcolo della differenza 
    differenzaMediaCharToken = abs(mediaCharTokenC2-mediaCharTokenC1)
    formattedOutput += createTable([[mediaCharTokenC1,mediaCharTokenC2,differenzaMediaCharToken]],[c1.getFileName(),c2.getFileName(),"Differenza"],["Media N caratteri per token"])+"\n\n"    
    
    #Calcolo numero dei lemmi distinti salvati in ordine alfabetico in un dizionario ed associati al nome del file
    c1.setLemmaList(lemmatizer(c1.getTokenList()))
    c2.setLemmaList(lemmatizer(c2.getTokenList()))
    formattedOutput += createTable([[len(c1.getLemmaList()),len(c2.getLemmaList())]], [c1.getFileName(),c2.getFileName()], ["Numero Lemmi Totali"])
    
    # calcolo la media dei lemmi per frase arrotondata a 3 cifre decimali
    mediaLemmiFrasiC1 = round(len(c1.getLemmaList())/len(c1.getSentenceList()),3)
    mediaLemmiFrasiC2 = round(len(c2.getLemmaList())/len(c2.getSentenceList()),3)
    formattedOutput += createTable([[mediaLemmiFrasiC1,mediaLemmiFrasiC2,abs(mediaLemmiFrasiC2-mediaLemmiFrasiC1)]], [c1.getFileName(),c2.getFileName(), "Differenza"], ["Media Lemmi per frase"])+"\n\n"
    
    # creazione della tabella che rappresenta la distribuzione POS nei primi 1000 tokens
    POSTable = POSDistribution(c1.getFileName(),c2.getFileName(), c1.getTokenList(), c2.getTokenList())
    formattedOutput += "DISTRIBUZIONE POS:\n"+POSTable+"\n\n"
    # print(formattedOutput)
    
    # calcolo della polarità per frase con algoritmo di sentiment analisys
    c1SentencePolarity = polaritySentenceClassification(c1.getSentenceList())
    c2SentencePolarity = polaritySentenceClassification(c2.getSentenceList())
    formattedOutput+= createTable([c1SentencePolarity.values(),c2SentencePolarity.values()], c1SentencePolarity.keys(),[c1.getFileName(),c2.getFileName()])
    # print(formattedOutput)
    
    # calcolo della polarità complessiva del documento, si poteva fare anche confrontando i risultati precedentemente ottenuti
    c1TextPolarity = polarityCorpusClassification(c1.getSentenceList())
    c2TextPolarity = polarityCorpusClassification(c2.getSentenceList())
    formattedOutput += f"\n\nPOLARITA' DEL DOCUMENTO:\nLa Polarità del documento {c1.getFileName()} e' {c1TextPolarity}.\nLa Polarita' del documento {c2.getFileName()} e' {c2TextPolarity}.\n"
    utils.writeFile(resultfilePath, formattedOutput)
    return
    
if __name__ == '__main__':
    #safe exit se non è stato passato nessun file
    if len(sys.argv)<3:
        print("Attenzione! Non hai passato nessun file da input.")
    else:
        main(sys.argv[1],sys.argv[2])