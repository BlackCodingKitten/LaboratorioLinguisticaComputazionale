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
import datetime


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
        print(f"Il file "+file_path+" inserito non è stato trovato")
        return "-1"
    except PermissionError:
        print(f"Non hai il permesso di aprire il file {file_path}")
        return "-1"
    except IOError:
        print(f"Errore generico di lettura del file {file_path}")
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
    """funzione che scrive l'output dell'esecuzione su un file .txt

    Args:
        tokens_lists (string list list): liste dei token del testo
        sentences_lists (string list list): liste delle frasi del testo
        types_lists (string list list): vocabolari dei testi
        files (string list ): nomi dei file
    """   
    max_types_n= ("",0) 
    max_tokens_n=("",0)
    max_sentences_n=("",0) 
    for i in range(0, len(files)):
        #controllo il counter maggiore delle frasi
        if(len(sentences_lists[i])>max_sentences_n[1]):
            max_sentences_n=(files[i],len(sentences_lists[i]))
        #controllo il counter maggiore dei token
        if(len(tokens_lists[i])>max_tokens_n[1]):
            max_tokens_n=(files[i],len(tokens_lists[i]))
        #controlllo il counter maggiore dei tipi
        if(len(types_lists[i])> max_types_n[1]):
             max_types_n=(files[i],len(types_lists[i]))
    
    result="RISULTATO DEL CONFRONTO TRA FILE:\nMax Sentences:\tIl file con il numero maggiore di frasi e' "+(max_sentences_n[0].split("/"))[-1]+" e contiene "+str(max_sentences_n[1])+" frasi.\n"+"Max Tokens:\t"+"Il file con il numero maggiore di token e' "+(max_tokens_n[0].split("/"))[-1]+" e contiene "+str(max_tokens_n[1])+" token.\n"+"Max Type:\t"+"Il file con il numero maggiore di tipi e' "+(max_types_n[0].split("/"))[-1]+" e contiene "+str(max_types_n[1])+" tipi.\n"
    
    with open("/home/mikela/Documents/LaboratoriLinguisticaComputazionale/LCEsercitazioni/ES 14.10.2024/results/results"+str(datetime.datetime.now())+".txt", "w") as results_file:
        results_file.write(result+"\n")
        for i in range(0,len(files)):
            string_to_write=("NOME DEL FILE:\t"+files[i]+"\n")
            string_to_write+="\nIl file contiene "+str(len(sentences_lists[i]))+" frasi, "+str(len(tokens_lists[i]))+" token e "+str(len(types_lists))+" tipi\n\n"
            string_to_write+="-"*15+"LISTA DELLE FRASI(ogni frase e' delimitata da una freccia):\n\n"+"\n-->".join(sentences_lists[i])+"\n"
            string_to_write+="-"*15+"LISTA DEI TOKEN:\n"+"\n".join(sorted(tokens_lists[i]))+"\n"
            string_to_write+="-"*15+"VOCABOLARIO DEI TIPI:\n"+"\n".join(sorted(types_lists[i]))+"\n"
            string_to_write+="\n\n\n"
            results_file.write(string_to_write)
        results_file.close()
        
        
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
            print(f"Impossibile leggere il file {path}")
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
    main(sys.argv[1:])
