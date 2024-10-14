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
import sys

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
        print("Il file {file_path} inserito non è stato trovato")
        return "-1"
    except PermissionError:
        print("Non hai il permesso di aprire il file {file_path}")
        return "-1"
    except IOError:
        print("Errore generico di lettura del file {file_path}")
        return "-1"



def tokens_extractor(file_contents):
    """funzione che prende in input il contenuto testuale di un file e tramite l'impiego di nltk tokenizer restituisce la lista dei token

    Args:
        file_contents (file): contenuto del file da tokenizzare

    Returns:
        list: lista dei token ch ecompongono il file 
    """    
    tokens_list = []
    sentences_list = nltk.tokenize.sent_tokenize(file_contents)
    for s in sentences_list:
        tokens_list.extend(nltk.tokenize.word_tokenize(s))
    return tokens_list



def types_extractor(tokens_list):
    """prende in input la lista dei token in un corpo e resituisce il suo vocabolario

    Args:
        tokens_list (string list): lista di tutti i token 

    Returns:
        list: vocabolario dei tipi della lista passata da input
    """    
    return list(set(tokens_list))
     


def main(argv_list): 
    #apro i file e ne leggo il contenuto
    file1 = read_file(argv_list[1])
    file2 = read_file(argv_list[2])
    
    if file1 == "-1" or file2 == "-1":
        return
    #creo le liste di tokens
    tokens_list1 = tokens_extractor(file1)
    tokens_list2 = tokens_extractor(file2)
    #creo i vocabolari
    types_list = types_extractor(tokens_list1)
    types_list = types_extractor(tokens_list2)
    
        
if __name__ == "__main__":
    main(sys.argv)#prende in input l'intero argv