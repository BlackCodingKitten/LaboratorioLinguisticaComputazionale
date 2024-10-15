"""ESERCIZIO DEL 14/10/2024
Scrivere un programma che:
• Stampi la divisione in frasi di ciascun file, e per ogni frase stampi il testo tokenizzato
• Stampi i token distinti (le parole tipo, o type) ordinati alfabeticamente contenuti all’interno di ciascun file
• Confronti tra loro i due file rispetto a:
    • numero di frasi
    • numero di token
    • numero di type
BONUS:
• Il programma deve scrivere l’output in un file
    • Redirezione delle stampe su stdout
    • <filename>.write(stringa)
        • Scrive la stringa stringa su <filename> (file aperto in modalità scrittura)
"""


import nltk
nltk.download('punkt_tab')
import sys
import json

def read_file(file_path):
    """funzione che prende in input il contenuto testuale di un file e tramite l'impiego di nltk tokenizer restituisce la lista dei token

    Args:
        file_path (string): percorso del file da leggere

    Returns:
        string: il contenuto del file o "-1" se si è verificata un'eccezionee 
    """ 
    try:
        file_pointer = open(file_path, "r")  
        return file_pointer.read() 
    except FileNotFoundError:
        print("Il file "+file_path+" inserito non è stato trovato")
        return "-1"
    except PermissionError:
        print("Non hai il permesso di aprire il file {file_path}")
        return "-1"
    except IOError:
        print("Errore generico di lettura del file {file_path}")
        return "-1"



def tokens_extractor(file_contents):
    """funzione che prende in input il contenuto testuale di un file e tramite l'impiego di nltk tokenizer restituisce la lista dei token, la lista delle frasi e il vocabolario dei tipi
    Args:
        file_contents (file): contenuto del file da tokenizzare

    Returns:
        list: lista dei token ch ecompongono il file, 
         lista delle frasi che compongono il file,
         vocabolario dei tipi 
    """    
    tokens_list = []
    sentences_list = nltk.tokenize.sent_tokenize(file_contents)
    for s in sentences_list:
        tokens_list.extend(nltk.tokenize.word_tokenize(s))
    return tokens_list, sentences_list, list(set(tokens_list))

     
def write_output(tokens_lists, sentences_lists, types_lists, files):
    """funzione che scrive l'output dell'esecuzione su un file json

    Args:
        tokens_lists (string list tuple): liste dei token del testo
        sentences_lists (string list tuple): liste delle frasi del testo
        types_lists (string list tuple): vocabolari dei testi
        files (string list tuple): nomi dei file
    """   
    max_types_n= ("",0) 
    max_tokens_n=("",0)
    max_sentences_n=("",0) 
    json_object_list=[]
    for i in range(0, len(files)):
        for k in range (0, len(sentences_lists[i])):
            sentences_lists[i][k]= str(i)+". "+sentences_lists[i][k]
        print(sentences_lists[i][k])
    #     json_object={
    #         {files[i]}:{
    #             "sentence list len":{len(sentences_lists[i])},
    #             "token list len":{len(tokens_lists[i])},
    #             "type list len":{len(types_lists[i])},
    #             "sentences list": {sentences_lists[i]},
    #             "token list": {tuple(sorted(tokens_lists[i]))},
    #             "type list":{tuple(sorted(types_lists[i]))}
    #         }
    #     }
    #     json_object_list.append(json.dumps(json_object, indent = 4))
    #     #controllo il counter m,aggiore delle frasi
    #     if(len(sentences_lists[i])>max_sentences_n):
    #         max_sentences_n=(files[i],len(sentences_lists[i]))
    #     #controllo il counter maggiore dei token
    #     if(len(tokens_lists[i])>max_tokens_n):
    #         max_tokens_n=(files[i],len(tokens_lists[i]))
    #     #controlllo il counter maggiore dei tipi
    #     if(len(types_lists[i])> max_types_n):
    #          max_types_n=(files[i],len(types_lists[i]))
    
    # result= {
    #     "Max Sentences": "Il file con il numero maggiore di frasi e' {max_sentences_n[0]} e contiene {max_sentences_n[1]} frasi.",
    #     "Max Tokens": "Il file con il numero maggiore di token e' {max_tokens_n[0]} e contiene {max_tokens_n[1]} frasi.",
    #     "Max Type": "Il file con il numero maggiore di tipi e' {max_types_n[0]} e contiene {max_types_n[1]} frasi."
    # }
    # json_object_list.append(json.dumps(result, indent=4))
    # with open('./results/results.json', "w") as results_file:
    #     for json_object in json_object_list:
    #         results_file.write(json_object)
    #     results_file.close()
        
        
def main(argv_list):
    file_list =[]
    valid_file_name =[]
    sentences_llist=[]
    types_llist=[]
    tokens_llist=[]
    #apro i file e ne leggo il contenuto
    for path in argv_list:
        file_text = read_file(path)
        if file_text != "-1":
            valid_file_name.append(path)
            file_list.append(file_text)
        else:
            print("Impossibile leggere il file {path}")
    if not file_list:
        print("Esecuzione terminata con errore.")
        return

    #tokenizzo il testo dei file in frasi, e parole, e restituisco il vocabolario dei tipi
    for file_text in file_list:
        t,s,ty = tokens_extractor(file_text)
        sentences_llist.append(s)
        tokens_llist.append(t)
        types_llist.append(ty)
  
    write_output(tokens_llist,sentences_llist,types_llist,valid_file_name)
    print("Esecuzione svolta correttemente troverai il file di output nella cartella \"results\".")

    return
    
        
if __name__ == "__main__":
    #prende solo i parametri effettivi (salta il nome del programma argv[0] e salta l'ultimo elemento che rappresenta la lunghezza di argv)
    #main(sys.argv[1::(int(sys.argv[len(sys.argv)-1])-1)])
    main(["/home/mikela/Documents/LaboratoriLinguisticaComputazionale/LCEsercitazioni/ES 14.10.2024/d1.txt","/home/mikela/Documents/LaboratoriLinguisticaComputazionale/LCEsercitazioni/ES 14.10.2024/d2.txt"])