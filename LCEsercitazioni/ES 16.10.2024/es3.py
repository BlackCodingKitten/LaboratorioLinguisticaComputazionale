""" 
• Scrivere un programma che, dato lo stesso testo dell’esercizio precedente, all’aumentare delle dimensioni testo (50 token per volta), stampi:
    • L’andamento della crescita lessicale del file
        • Crescita non lineare del vocabolario
    • L’andamento della distribuzione degli hapax
"""

import nltk
import sys
import datetime

def read_file(path):
    try:
        file_ptr = open(path, "r")
        text = file_ptr.read()
        file_ptr.close()
        return text
    except IOError or FileNotFoundError:
        return False


def tokenizer (text):
    tokens = []
    for s in nltk.tokenize.sent_tokenize(text):
        tokens.extend(nltk.tokenize.word_tokenize(s))
    return tokens

def output_string_generator(tokens):
    """funzione che prende in input la lista dei token calcola le metriche del file e ritorna l'output formattato pronto da stampare

    Args:
        tokens (str list): listad ei tokens tokenizzati nel testo

    Returns:
        str: stringa con i risultati da stampare
    """    
    to_print=f"Il testo contiene {len(tokens)} parole token totali, le parole tipo totali sono {len(set(tokens))}.\n\n"
    V = len(set(tokens))
    C = len(tokens)
    n_type_prec = 0
    index = 0
    for i in range (50,C,50):        
        to_print+=f"{index+1+(i-50)}.) Nei primi {i} token la TTR è del {round((len(set(tokens[0:i]))/len(tokens[0:i]))*100,3)} %,\n"
        to_print+=f"sono state individuate {len(set(tokens[0:i]))-n_type_prec} parole tipo nuove,\n"
        to_print+=f"nei primi {i} token troviamo il {round((len(set(tokens[0:i]))/V)*100, 3)} % del vocabolario totale,\nla dimensione del vocabolario è cresciuta del {round(((len(set(tokens[0:i]))-n_type_prec)/V)*100,3)} %.\n\n"
        n_type_prec = len(set(tokens[0:i]))
    
    # Ultima sezione della lista che viene esclusa dallo slicing
    to_print+=f"{index+1}.) Nei primi {C} token la TTR è del {round((len(set(tokens[0:C]))/C)*100,3)} %,\n"
    to_print+=f"sono state individuate {len(set(tokens[0:C]))-n_type_prec} parole tipo nuove,\n"
    to_print+=f"nei primi {C} token troviamo il {round((len(set(tokens[0:C]))/V)*100, 3)} % del vocabolario totale,\nla dimensione del vocabolario è cresciuta del {round(((len(set(tokens[0:C]))-n_type_prec)/V)*100,3)} %.\n\n"
    return to_print
    


def main (argv):
    #safe exit se non è stato passato nessun file
    if len(argv)<2:
        print("Attenzione! Non hai passato nessun file da input.")
        return
    
    file_path = argv[1]
    #leggo il contenuto del file
    file_content = read_file(file_path)
    if not file_content:
        print("Errore nella lettura del contenuto del file. END")
        return
    #estraggo la lista dei token non ordinata
    tokens = tokenizer(file_content)
    print(output_string_generator(tokens))
        
    


if __name__ == '__main__':
    main(sys.argv)