import nltk
from matplotlib import pyplot as plt
import re
from collections import OrderedDict

text = "Ogni individuo ha diritto alla vita, alla libertà ed alla sicurezza della propria persona. Nessun individuo potrà essere tenuto in stato di schiavitù o di servitù; la schiavitù e la tratta degli schiavi saranno proibite sotto qualsiasi forma."
path="/mnt/c/Users/USER/Desktop/testo_tokenizzato.txt"
def tokenizer_to_text(testo, maiuscole_flag=False):
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
    with open("/mnt/c/Users/USER/Desktop/testo_tokenizzato.txt", "w")  as testo_tokenizzato:
        testo_tokenizzato.write(("\n").join(lista))
        testo_tokenizzato.close()
    print(lista)
    return 

def from_file_to_tokens(path):
    l= []
    with open("/mnt/c/Users/USER/Desktop/testo_tokenizzato.txt", "r") as ttk:
        token = ttk.readline()
        while token:
            if(token[-1]=="\n"):
                # print("-"*20+"DEBUG:l'ultimo carattere dei token è l'accapo, elimino con lo slicing")
                token=token[0:(len(token)-1)]
            l.append(token)
            token = ttk.readline()
    # print(l)
    return l

def TTR(tokens):
    print("LUNGHEZZA VOCABOLARIO DELLE PAROLE TIPO: ", len(vocabolario(tokens)))
    print("LUNGHEZZA LISTA DEI TOKEN", len(tokens))
    return round(len(vocabolario(tokens))/len(tokens), 3)
    
def vocabolario(tokens):
    pt = sorted(set(tokens))
    V = {}
    for t in pt:
        #calcola le frequenze assolute
        if t not in pt.keys():
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
    f=dict(fC)
    for i,v in f.items():
        print(f"La classe {i} contiene {len(v)} parole")
    
    return f


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

def freq_Cumulata(frequencyClassDic, C):
    s = 0
    lista = []
    frequencyClassDic = dict(OrderedDict(sorted(frequencyClassDic.items())))
    for k in frequencyClassDic:
        s += ((k)*(len(frequencyClassDic[k])))/C
        lista.append(f"La classe di frequenza {k} contiene {len(frequencyClassDic[k])} parole token, la frequenza cumulata è {round(s,4)*100}%\n") 
    return lista



# tokenizer_to_text(text,True)
# grafico_spettro_di_frequenza(frequencyClass(vocabolario(from_file_to_tokens(path))))
# somma=0
# for i in vocabolario(from_file_to_tokens(path)).values():
#     somma+=i
# print(somma/35)
# frequencyClass(vocabolario(from_file_to_tokens(path)))
print(freq_Cumulata(frequencyClass(vocabolario(from_file_to_tokens(path))),len(from_file_to_tokens(path))))