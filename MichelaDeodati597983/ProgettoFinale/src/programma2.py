"""Programma 2"""

import nltk
import sys
import utils
import math
from Corpus import Corpus
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from nltk.corpus import stopwords 
 
# nltk.download('stopwords')

"""
4. I top-10 bigrammi compos> da Verbo e Sostan>vo, ordina> per: 
    d. MI (Mutual Informa>on) massima, e rela>vo valore di MI 
    e. LMI (Local Mutual Informa>on) massima, e rela>vo valore di MI 
    f. Calcolare e stampare il numero di elemen> comuni ai top-10 per MI e per LMI
"""

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
    # Ordina le liste per ogni tag POS in base alla frequenza, in ordine decrescente in modo da averli già ordinati per quando li necessito dopo
    for y in result_dict:
        result_dict[y] = sorted(result_dict[y], key=lambda t: t[1], reverse=True)
    # Restituisce il dizionario ordinato
    return result_dict

def ngramsPOSTagging(tokens,ngramsDim):
    #mi creo il pos tagging dei token
    posList = nltk.tag.pos_tag(tokens, tagset='universal')
    #raggruppo le tuple in ngrammi di dimensione passata
    ngramsTokenPOS = list(nltk.ngrams(posList, ngramsDim))
    #estraggo la lista di ngrammi di soli POS tag:
    ngramsPOSList = []
    for elem in ngramsTokenPOS:
        # elem è la tupla che contiene le tupla (token,TAG)
        tagList = []
        for x in elem:
            # mi creo la lista di tutti i tag, ordinati per comparsa
            tagList.append(x[1])
        #converto la lista appena generata in una tupla e la inserisco nella lista degli ngrammi di soli tag
        ngramsPOSList.append(tuple(tagList))
    return getTop20Ngrams(ngramsPOSList)
            
def getTop20Ngrams(lista, l = 20):
    """conta la frequenza degli ngrammi contenuti nella lista

    Args:
        lista (list): lista di ngrammi

    Returns:
        dict: dizionario dei primi 20 ngrammi ordinato per frequenza discendente 
    """
    #Counter lista genrale la lista di tuple che ha come primo elemnto l'elemento della lista e come secondo il suo counter, l'ordinamento 
    # viene fatto con la funzione sorted sulla base del secondo valore x: x[1], e il reverse=True è per avere l'ordinamento decrescente
    return dict(sorted(dict(Counter(lista)).items(), key=lambda x: x[1], reverse=True)[:l])
    
def stopwordsDistribution(token):
    """prende la lista dei token di un corpus, trasforma tutto in minuscole, e calcola la distribuzione delle stopwords rispetto al totale dei token 

    Args:
        token (list): lista dei tokens di un corpus

    Returns:
        float: ditribuzione delle stopwords, arrotondata alla terza cifra decimale
    """   
    #Normalizza le maiuscole 
    tokens = [t.lower() for t in token]
    count=0 # contatore delle stopwords
    # ciclo for per contare le stopwords presenti nel testo
    for sw in stopwords.words('english'):
        count+= tokens.count(sw)
    # ritorno la distribuzione di frequenza
    return round(count/len(tokens), 3)

def dictBigramsTag(tokens):
    #mi creo il pos tagging dei token
    posList = nltk.tag.pos_tag(tokens, tagset='universal')
    #raggruppo le tuple in bigrammi
    bigramTokenPOS = list(nltk.ngrams(posList, 2))
    bigramDict = {}
    for ((t1,tag1),(t2,tag2)) in bigramTokenPOS:
        if bigramDict.get((tag1,tag2)) == None:
            bigramDict[(tag1,tag2)] = [(t1,t2)]
        else:
            bigramDict[(tag1,tag2)].append((t1,t2))

    NV = dict(sorted(dict(Counter(bigramDict[('NOUN','VERB')])).items(), key=lambda x: x[1], reverse=True))
    VN = dict(sorted(dict(Counter(bigramDict[('VERB', 'NOUN')])).items(), key=lambda x: x[1], reverse=True))
    return VN,NV

def jointProbability(tokens, conditionalProbabilityDict, bigramDicts):
    # P(t1,t2) = P(t2|t1) * P(t1)
    jointProbabilityDict = {}
    for (a,b) in bigramDicts.keys():
        jointProbabilityDict[(a,b)] = conditionalProbabilityDict[(a,b)] * (tokens.count(a)/len(tokens))
    return jointProbabilityDict

def conditionalProbability(tokens,bigramDict):
    # P(t1|t2) = freq(t1,t2) / freq(t1)
    bigrams= list(nltk.bigrams(tokens))
    conditionalProbabilityDict = {}
    for (a,b) in bigramDict.keys():
        conditionalProbabilityDict[(a,b)] =  bigrams.count((a,b)) / tokens.count(a) 
    return conditionalProbabilityDict

def MI(tokens, bigramsDict):
    C = len(tokens)
    MIDict = {}
    for (a,b),bfreq in bigramsDict.items():
        MIDict[(a,b)] = math.log2((bfreq*C)/((tokens.count(a))*(tokens.count(b))))
    return MIDict

def LMI(MIDict, bigramsDic):
    LMIDict = {}
    for key,v in MIDict.items():
        LMIDict[key] = v*bigramsDic[key]
    return LMIDict

def commonElem(MIDict,LMIDict):
    common = {}
    for key,value in MIDict.items():
        LMIValue = LMIDict.get(key)
        if LMIValue != None:
            common[key] = (value,LMIValue)
    return common

def main(filePath):
    #inzializzazione dell'istanza per il corpus
    corpus=Corpus(filePath)
    #leggo il file
    corpus.setText(utils.readFile(filePath))
    #creo la lista di token (non ordinata)
    corpus.setTokenList(utils.tokenSplitter(corpus.getText()))
    #creo la lista di frasi
    corpus.setSentenceList(utils.sentenceSplitter(corpus.getText()))
    #creo il vocabolario delle parole tipo
    corpus.setVocabulary()
    #inizializzo la stringa che verrà poi scritta nell'output e conterrà il file formattato
    formattedOutput = ''
    
    #Creazione del dizionario che ha come chiave i tag (tagset='universal') e come valori le liste di tuple contenenti come primo elemento i token associati a quel tag e come secondo elemento la frequenza
    POSDict = dict(POSTagging(corpus.getTokenList()))
    #i tag da ricercare sono: ADJ-> aggettivi  NOUN->sostantivi  #ADV->avverbi
    # estraggo i primi 50 aggettivi più frequenti dal dizionario
    top50ADJDict= dict(POSDict.get("ADJ")[:50])
    # estraggo i primi 50 nomi più frequenti dal dizionario
    top50NOUNDict= dict(POSDict.get("NOUN")[:50])
    # estraggo i primi 50 avverbi più frequenti dal dizionario
    top50ADVDict= dict(POSDict.get("ADV")[:50])
    #formatto l'output
    formattedOutput += f"\nTOP 50 AGGETTIVI PRESENTI NEL FILE {corpus.getFileName()}" + utils.createTable(top50ADJDict.values(), ['Frequenza'], top50ADJDict.keys())+"\n"
    formattedOutput += f"\nTOP 50 SOSTANTIVI PRESENTI NEL FILE {corpus.getFileName()}" + utils.createTable(top50NOUNDict.values(), ['Frequenza'], top50NOUNDict.keys())+"\n"
    formattedOutput += f"\nTOP 50 AVVERBI PRESENTI NEL FILE {corpus.getFileName()}" + utils.createTable(top50ADVDict.values(), ['Frequenza'], top50ADVDict.keys())+"\n"
    
    # creo la lista di tutti gli ngrammi [lista_monogrammi,lista_bigrammi,lista_trigrammi]:
    ngramsLList = [list(nltk.ngrams(corpus.getTokenList(),1)),list(nltk.ngrams(corpus.getTokenList(),2)),list(nltk.ngrams(corpus.getTokenList(),3))]
    # Prendo i primi 20 ordinati per frequenza
    top20Monograms = getTop20Ngrams(ngramsLList[0])
    top20Bigrams = getTop20Ngrams(ngramsLList[1])
    top20Trigrams = getTop20Ngrams(ngramsLList[2])
    # formatto loutput
    formattedOutput += f"\nTOP 20 MONOGRAMMI PRESENTI NEL FILE {corpus.getFileName()}:\n"+ utils.createTable(top20Monograms.values(), ['Frequenza'], top20Monograms.keys())+"\n"
    formattedOutput += f"\nTOP 20 BIGRAMMI PRESENTI NEL FILE {corpus.getFileName()}:\n"+ utils.createTable(top20Bigrams.values(), ['Frequenza'], utils.tupleListToString(top20Bigrams.keys()))+"\n"
    formattedOutput += f"\nTOP 20 TRIGRAMMI PRESENTI NEL FILE {corpus.getFileName()}:\n"+ utils.createTable(top20Trigrams.values(), ['Frequenza'], utils.tupleListToString(top20Trigrams.keys()))+"\n"
    # Calcolo distribuzione delle stopwords sul testo e formattazione nel file di output
    formattedOutput += f"\n\nLa percentuale di Stopwords presenti nel testo {corpus.getFileName()} e' {stopwordsDistribution(corpus.getTokenList()) * 100}%\n"
    
    # calcolo distribuzione dei pronomi e media per frase #PRON -> pronomi 
    totalePronomi = sum([x for y,x in POSDict.get('PRON')])
    # formatto l'output inserendo il rapporto tra il numero di pronomi e il totlae dei token ricavato grazie alla lista non ordinata di token contenuta nell'istanza "corpus"
    # Calcolo anche la media di pronomi per frasi grazie alla lista non ordinata di frasi contenuta nell'istanza "corpus"
    formattedOutput +=f"\nNel corpus {corpus.getFileName()} ci sono {totalePronomi} pronimi in totale, in rapporto al totale dei token equivale al {round((totalePronomi/len(corpus.getTokenList()))*100,3)}%, per ogni frase ci sono in media, circa, {round(totalePronomi/len(corpus.getSentenceList()),3)} pronomi.\n\n"
    
    #creo la lista contenente gli ngrammi di tag con relativa frequenza abbinata di dimensione 1,2,3,4,5
    ngramsTagLList = [ngramsPOSTagging(corpus.getTokenList(),1),ngramsPOSTagging(corpus.getTokenList(),2),ngramsPOSTagging(corpus.getTokenList(),3),ngramsPOSTagging(corpus.getTokenList(),4),ngramsPOSTagging(corpus.getTokenList(),5)]
    # formatto l'output
    # print(ngramsTagLList[1].keys())
    for index in range(5):
        formattedOutput += f"\n\nTOP 20 {index+1}-GRAMMI PRESENTI NEL FILE {corpus.getFileName()}:\n"+utils.createTable(ngramsTagLList[index].values(),['Frequenza'], utils. tupleListToString(ngramsTagLList[index].keys()))+"\n"

    #Punto 4-> i bigrammi Verbo-Sostantivo quindi sia VERB -> verbo NOUN->SOSTANTIVO (VERB,NOUN) (NOUN,VERB)
    #Creo un dizionario che contiene tutti e soli i bigrammi composti da Nome-Verbo e da Verbo nome con associata relativa frequenza
    verboNome,nomeVerbo = dictBigramsTag(corpus.getTokenList())
    formattedOutput += f"\n\nTOP 10 BIGRAMMI (VERBO-SOSTANTIVO) ORDIANTI PER FREQUENZA ASSOLUTA:\n" + utils.createTable([(x,x/len(corpus.getTokenList())) for x in list(verboNome.values())[:10]], ["Frequenza Assoluta","Frequenza Relativa"], list(utils.tupleListToString(verboNome.keys()))[:10])+"\n"
    formattedOutput += f"\n\nTOP 10 BIGRAMMI (SOSTANTIVO-VERBO) ORDIANTI PER FREQUENZA ASSOLUTA:\n" + utils.createTable([(x,x/len(corpus.getTokenList())) for x in list(nomeVerbo.values())[:10]], ["Frequenza Assoluta","Frequenza Relativa"], list(utils.tupleListToString(nomeVerbo.keys()))[:10])+"\n"
    
    #conditionalProbability 
    verboNomeconditionalProbability = conditionalProbability(corpus.getTokenList(), verboNome)
    nomeVerboconditionalProbability = conditionalProbability(corpus.getTokenList(), nomeVerbo)
    formattedOutput+=f"\n\nTOP 10 BIGRAMMI (VERBO-SOSTANTIVO) ORDIANTI PER CONDITIONAL PROBABILITY:\n" +utils.createTable(getTop20Ngrams(verboNomeconditionalProbability,10).values(), ["Probabilita' Condizionata"], getTop20Ngrams(verboNomeconditionalProbability,10).keys())
    formattedOutput+=f"\n\nTOP 10 BIGRAMMI (SOSTANTIVO-VERBO) ORDIANTI PER CONDITIONAL PROBABILITY:\n" +utils.createTable(getTop20Ngrams(nomeVerboconditionalProbability,10).values(), ["Probabilita' Condizionata"], getTop20Ngrams(nomeVerboconditionalProbability,10).keys())
    
    #jointProbability
    verboNomejointProbability = jointProbability(corpus.getTokenList(), verboNomeconditionalProbability, verboNome)
    nomeVerbojointProbability = jointProbability(corpus.getTokenList(), nomeVerboconditionalProbability, nomeVerbo)
    formattedOutput += f"\n\nTOP 10 BIGRAMMI (VERBO-SOSTANTIVO) ORDIANTI PER JOINT PROBABILITY:\n" + utils.createTable(getTop20Ngrams(verboNomejointProbability,10).values(), ['Probabilita\' Congiunta'], utils.tupleListToString(getTop20Ngrams(verboNomejointProbability,10).keys()))
    formattedOutput += f"\n\nTOP 10 BIGRAMMI (SOSTANTIVO-VERBO) ORDIANTI PER JOINT PROBABILITY:\n" + utils.createTable(getTop20Ngrams(nomeVerbojointProbability,10).values(), ['Probabilita\' Congiunta'], utils.tupleListToString(getTop20Ngrams(nomeVerbojointProbability,10).keys()))

    #mutual Information
    verboNomeMI = MI(corpus.getTokenList(),verboNome)
    nomeVerboMI = MI(corpus.getTokenList(), nomeVerbo)
    verboNomeLMI = LMI(verboNomeMI,verboNome)
    nomeVerboLMI = LMI(nomeVerboMI, nomeVerbo)
    verboNomeCommon = dict(commonElem(getTop20Ngrams(verboNomeMI,10),getTop20Ngrams(verboNomeLMI,10)))
    nomeVerboCommon = dict(commonElem(getTop20Ngrams(nomeVerboMI,10), getTop20Ngrams(nomeVerboLMI,10)))
    
    formattedOutput += f"\n\nTOP 10 BIGRAMMI (VERBO-SOSTANTIVO) MI:\n"+utils.createTable(getTop20Ngrams(verboNomeMI,10).values(), ["Mutual Information"], utils.tupleListToString(getTop20Ngrams(verboNomeMI,10).keys()))
    formattedOutput += f"\n\nTOP 10 BIGRAMMI (SOSTANTIVO-VERBO) MI:\n"+utils.createTable(getTop20Ngrams(nomeVerboMI,10).values(), ["Mutual Information"], utils.tupleListToString(getTop20Ngrams(nomeVerboMI,10).keys()))
    formattedOutput += f"\n\nTOP 10 BIGRAMMI (VERBO-SOSTANTIVO) LMI:\n"+utils.createTable(getTop20Ngrams(verboNomeLMI,10).values(), ["Mutual Information"], utils.tupleListToString(getTop20Ngrams(verboNomeLMI,10).keys()))
    formattedOutput += f"\n\nTOP 10 BIGRAMMI (SOSTANTIVO-VERBO) LMI:\n"+utils.createTable(getTop20Ngrams(nomeVerboLMI,10).values(), ["Mutual Information"], utils.tupleListToString(getTop20Ngrams(nomeVerboLMI,10).keys()))
    formattedOutput += f"\n\nTOP 10 BIGRAMMI (VERBO-SOSTANTIVO) IN COMUNE TRA MI E LMI:\n"+utils.createTable(verboNomeCommon.values() ,["MI","LMI"], utils.tupleListToString(verboNomeCommon.keys()))
    formattedOutput += f"\n\nTOP 10 BIGRAMMI (SOSTANTIVO-VERBO) IN COMUNE TRA MI E LMI:\n"+utils.createTable(nomeVerboCommon.values(), ["MI","LMI"], utils.tupleListToString(nomeVerboCommon.keys()))
    
    
    
    print(formattedOutput)
    return

if __name__ == '__main__':
    #safe exit se non è stato passato nessun file
    if len(sys.argv)<2:
        print("Attenzione! Non hai passato nessun file da input.")
    else:
        main(sys.argv[1])
    