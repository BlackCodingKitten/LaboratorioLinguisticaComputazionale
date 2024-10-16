"""
• Utlizzando Python e le funzioni di NLTK, scrivere un programma che:
    • stampi gli hapax (token con frequenza 1) del testo
    • stampi la distribuzione di hapax nel testo: |𝑉1|/|𝐶|
    • ottenga il vocabolario del testo ordinato alfabeticamente
        • Per ogni elemento del vocabolario, ne stampi frequenza assoluta e relativa nel file
    • stampi la Type-Token Ratio del testo
        • 𝑇𝑇𝑅 = |𝑉𝐶|/|𝐶|
"""

import datetime
import nltk
# nltk.download('punkt_tab')
import sys

def read_file(path):
    """funzione che prende in input il percorso di un file e ne restituisce il contenuto come stringa 

    Args:
        path (_str_): percorso del file

    Returns:
        str : contenuto testuale del file
    """    
    try:
        file = open(path, "r")
        text = file.read()
        file.close()
        return text
    except IOError or FileNotFoundError:
        return False

def tokenizer (text):
    """funzione che prende in input un testo e restitusce la lista delle parole token che lo compongono

    Args:
        text (_str_): stringa da tokenizzare

    Returns:
        list: lista ordinata alfabeticamente dei token
    """    
    t_list = []
    s_list = nltk.tokenize.sent_tokenize(text)
    for s in s_list:
        t_list.extend(nltk.tokenize.word_tokenize(s))
    return sorted(t_list)


def main(argv):
    
    file_content = read_file(argv[1])
    #controllo che non ci siano stati errori nella lettura del file
    if not file_content:
        print("Errore lettura del contenutop del file.")
        return
    
    tokens = tokenizer(file_content)
    #calcolo la dimensione della lista delle parole token
    C = len(tokens) 
    
    #uso un dizionario per salvare le parole tipo come chiavi e asseganre a ciascuna il valore della frequenza assoluta e della frequenza relativa
    frequency_abs_rel = {}
    
    for type in sorted(set(tokens)):
        fa = tokens.count(type)
        fr = fa/C
        frequency_abs_rel[type] = (fa,round(fr,3)) 
    
    #calcolo il type-token ratio
    TTR = round(len(frequency_abs_rel.keys()) / C, 3)
    
    #inserisco in una stringa tutto quello che devo stampare 
    to_print = "TYPE TOKEN RATIO = "+str(TTR)+"\n\n"
    to_print += "|PAROLE TIPO|\t|FREQUENZA ASSOLUTA|\t|FREQUENZA RELATIVA|\n"
    hapax_list =[]
    for type in frequency_abs_rel.keys():
        (absolute, relative)=frequency_abs_rel[type]
        to_print+=type+":"+" "*(22-len(type))+"\t"+str(absolute)+"\t\t\t\t\t\t"+str(relative)+"\n"
        if(absolute == 1):
            hapax_list.append(type)
            
    to_print+="\n"+"-"*13+"LISTA DEGLI HAPAX:\n"+str(hapax_list)
 
    #scrivo l'output su un file
    with open("/home/mikela/Documents/LaboratorioLinguisticaComputazionale/LCEsercitazioni/ES XX.10.2024/output_file_"+((argv[1].split("/"))[-1]).split(".")[0]+str(datetime.datetime.now())+".txt", "w") as output_file:
        output_file.write(to_print)
    output_file.close()
    return 

if __name__ == '__main__':
    main(sys.argv)