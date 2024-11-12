# A partire dal testo tokenizzato Costituzione_token.txt:
# (NB: potete usare python, javascript, excel, NLTK, comandi della shell di linux)

# - calcolate la lunghezza del corpus;
# - calcolate il vocabolario;
# - calcolate la TTR;
# - calcolate le classi di frequenza e costruite lo spettro di frequenza;
# - calcolate la distribuzione di frequenza e ordinatela per frequenza discendente;
# - costruite la distribuzione di Zipf (rango/frequenza);
# - calcolate la frequenza cumulata relativa di parole token

from matplotlib import pyplot as plt
from collections import OrderedDict
import numpy

def readFile(path):
    Vocabulary = {}
    keyList = []
    with open(path, "r") as filePtr:
        for token in filePtr:
            keyList.append(token.strip())
        filePtr.close()
        for token in keyList:
            if token not in Vocabulary.keys():
                Vocabulary[token] = 1
            else:
                Vocabulary[token] +=1
    return Vocabulary,keyList

def frequencyClass (vocabulary):
    fC = {}
    for fClass in vocabulary.values():
        if fClass not in fC.keys():
            fC[fClass] = [k for k,v in vocabulary.items() if v==fClass]
    return fC
            
    


def main (filePath):
    toPrint = ""
    vocabulary, tokens = readFile(filePath)
    toPrint += f"Il Corpus è lungo in totale {len(tokens)} parole token.\n"
    toPrint += f"Il Vocabolario è lungo in totale {len(vocabulary.keys())} parole tipo.\n"
    toPrint += f"Il TTR (Type Token Ratio) del testo è {len(vocabulary.keys())/len(tokens)}\n\n"
    frequencyClassDic = dict(sorted((frequencyClass(vocabulary)).items()))
    for k,v in frequencyClassDic.items():
        toPrint += f"La classe di frequeza {k} contine {len(v)} parole, e sono: {v}\n"
    x = list(frequencyClassDic.keys())
    y = [len(v) for v in frequencyClassDic.values()]
    # print(x)
    # print(y)
    plt.plot(x,y,marker='o', linestyle='-')
    plt.title("Spettro di Frequenze ")
    plt.xlabel("X =Indice Classe di Frequenza")
    plt.ylabel("Y = Numerosità della Classe")
    plt.xticks(x)
    plt.grid(True)
    plt.savefig("/home/mikela/Documents/LaboratorioLinguisticaComputazionale/LCEsercitazioni/Esercitazione Lenci 28/ES 25.10.2024/graficoSpettroDiFrequenza.svg")
    freqDesc = dict(OrderedDict(reversed(sorted(vocabulary.items(), key=lambda x:x[1]))))#ordina sulla base dei valori
    # print(freqDesc)
    toPrint += "\n\nPAROLE\t\t\t\t\t\tFREQUENZE ASSOLUTE\t\tFREQUENZE RELATIVE\n\n"
    for k,v in vocabulary.items():
        toPrint += f"{k}\t\t\t\t\t\t{v}\t\t\t\t\t{round(v/len(tokens),3)}\n"
        
    rango = {}
    r = 0
    frequenzaMax = max(freqDesc.values())
    for k,v in freqDesc.items():
            rango[r] = (k,v)
            r+=1
            
    #a questo punto ho tutte le parole ordinate per rango 
    plt.cla()
    plt.clf()
    plt.close()
    
    plt.scatter([x for x in rango.keys()],[t[1] for t in rango.values()],color='b',alpha=0.7)
    plt.xscale('log')
    plt.grid(True)   
    plt.yscale('log')
    plt.xlabel("X=Rango")
    plt.ylabel("Y=Frequenza")
    plt.title('Distribuzione di Zipf')
    plt.savefig("/home/mikela/Documents/LaboratorioLinguisticaComputazionale/LCEsercitazioni/Esercitazione Lenci 28/ES 25.10.2024/graficoZipf.svg")
    # for i in range(0,r,1):
    plt.cla()
    plt.clf()
    plt.close()
    # print(rango)      
    print(toPrint)
    
    return




if __name__=='__main__':
    main("/home/mikela/Documents/LaboratorioLinguisticaComputazionale/LCEsercitazioni/Esercitazione Lenci 28/ES 25.10.2024/Costituzione_token.txt")