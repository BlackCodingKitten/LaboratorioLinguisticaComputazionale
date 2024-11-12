import nltk
import sys
from matplotlib import pyplot as plt
import re
from collections import OrderedDict

text = "Se riesci a tenere la testa a posto quando tutti intorno a te l'hanno persa e danno la colpa a te, se puoi avere fiducia in te stesso quando tutti dubitano di te, ma prendi in considerazione anche i loro dubbi."
# text ="Era il tempo migliore e il tempo peggiore, la stagione della saggezza e la stagione della follia, l’epoca della fede e l’epoca dell’incredulità, il periodo della luce e il periodo delle tenebre, la primavera della speranza e l’inverno della disperazione."


def tokenizer(testo, maiuscole_flag=False):
    regex = "[,:\.!?;-]+"
    regex2 = "[']"
    if(maiuscole_flag):
        print("-"*20+"DEBUG: normalizzo le maiuscole")
        testo=testo.lower()
    lista = nltk.tokenize.word_tokenize(testo)
    #correzione dell'apostrofo
    for t in lista:
        if "\'" in t:
            print("-"*20+"DEBUG: ho trovato un apostrofo e ho fatto lo split")
            lista.extend(t.split("\'"))
            lista.remove(t)
    return lista

def TTR(tokens):
    print("LUNGHEZZA VOCABOLARIO DELLE PAROLE TIPO: ", len(vocabolario(tokens)))
    print("LUNGHEZZA LISTA DEI TOKEN", len(tokens))
    return round(len(vocabolario(tokens))/len(tokens), 3)
    
def vocabolario(tokens):
    pt = sorted(set(tokens))
    V = {}
    for t in pt:
        #calcola le frequenze assolute
        V[t]=tokens.count(t)
    return V

def vocabolario_frel(tokens):
    C=len(tokens)
    V ={}
    for k,v in vocabolario(tokens):
         V[k]=v/C
    return V
    
  
def frequencyClass(vocabulary):
    fC = {}
    for fClass in vocabulary.values():
        if fClass not in fC.keys():
            fC[fClass] = [k for k,v in vocabulary.items() if v==fClass]
    return dict(fC)


def grafico_spettro_di_frequenza(frequencyClassDic):
    frequencyClassDic = dict(sorted(frequencyClassDic.items()))
    plt.cla()
    plt.clf()
    plt.close()
    x = list(frequencyClassDic.keys())
    y = [len(v) for v in frequencyClassDic.values()]
    plt.plot(x,y,marker='o', linestyle='-')
    plt.title("Spettro di Frequenza")
    plt.xlabel("X = Indice Classe di Frequenza")
    plt.ylabel("Y = Numerosità della Classe")
    plt.xticks(x)
    plt.grid(True)
    plt.savefig("/mnt/c/Users/USER/Desktop/grafico_spettro_di_frequenza.png")
    print("-"*20+"DEBUG: grafico dello spettro fi frequenze calcolato correttamente")

def graficoZipf(V):
    freqDesc = dict(OrderedDict(reversed(sorted(V.items(), key=lambda x:x[1]))))
    rango = {}
    r=1
    C= max(freqDesc.values())
    for k,v in freqDesc.items():
        rango[r]=(k,v)
        r+=1
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
    plt.savefig("/mnt/c/Users/USER/Desktop/Zipf.png")
    return rango

# # #scrive un file sul desktop con il testo tokenizzato, prima flag rimuove le maiuscole seconda flag rimuove la punteggiatura
# # with open("/mnt/c/Users/USER/Desktop/testo_tokenizzato.txt", "w") as testo_tokenizzato:
# #     # testo_tokenizzato.write(("\n").join(tokenizer("questo è UN TESTO di prova, per vedere! se me lo mette: sul desktop. \n",False,False)))  
# #     # testo_tokenizzato.write(str(vocabolario(tokenizer("questo è UN TESTO testo di prova, per vedere! se me lo mette: sul desktop. \n",True,False)))) 
# #     testo_tokenizzato.close()

# print(tokenizer(text))
# print("IL TTR E': ",TTR(tokenizer(text,True)))

def freq_Cumulata(frequencyClassDic, C):
    s = 0
    lista = []
    frequencyClassDic = dict(OrderedDict(sorted(frequencyClassDic.items())))
    for k in frequencyClassDic:
        s += ((k)*(len(frequencyClassDic[k])))/C
        lista.append(f"La classe di frequenza {k} contiene {len(frequencyClassDic[k])} parole token, la frequenza cumulata è {round(s,4)}\n") 
    return lista

print(graficoZipf(vocabolario(tokenizer(text,True))))

# for i in freq_Cumulata(frequencyClass(vocabolario(tokenizer(text, True))),len(tokenizer(text, True))):
#     print(i)