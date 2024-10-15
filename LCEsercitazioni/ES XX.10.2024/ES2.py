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
    """_summary_

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """    
    try:
        file = open(path, "r")
        text = file.read()
        file.close()
        return text
    except IOError or FileNotFoundError:
        return False

def tokenizer (text):
    """_summary_

    Args:
        text (_type_): _description_

    Returns:
        _type_: _description_
    """    
    t_list = []
    s_list = nltk.tokenize.sent_tokenize(text)
    for s in s_list:
        t_list.extend(nltk.tokenize.word_tokenize(s))
    return sorted(t_list)


def main(argv):
    
    file_content = read_file(argv[1])
    
    if not file_content:
        print("Errore lettura del contenutop del file.")
        return
    
    tokens = tokenizer(file_content)
    C = len(tokens)
    
    frequency_abs_rel = {}
    
    for type in sorted(set(tokens)):
        fa = tokens.count(type)
        fr = fa/C
        frequency_abs_rel[type] = (fa,round(fr,3)) 
    
    TTR = round(len(frequency_abs_rel.keys()) / C, 3)
    
    to_print = "TYPE TOKEN RATIO = "+str(TTR)+"\n"
    to_print = "-"*20+"LISTA DEI TIPI CON |FREQUENZA ASSOLUTA|\t|FREQUENZA RELATIVA|\n"
    hapax_list =[]
    for type in frequency_abs_rel.keys():
        (absolute, relative)=frequency_abs_rel[type]
        to_print+=type+":\t|"+str(absolute)+"|\t|"+str(relative)+"|\n"
        if(absolute == 1):
            hapax_list.append(type)
            
    to_print+="\n"+"-"*20+"LISTA DEGLI HAPAX:\n"+"\n".join(sorted(hapax_list))
    
    with open("/home/mikela/Documents/LaboratorioLinguisticaComputazionale/LCEsercitazioni/ES XX.10.2024/output_file"+str(datetime.datetime.now())+".txt", "w") as output_file:
        output_file.write(to_print)
    output_file.close()
    return 

if __name__ == '__main__':
    main(sys.argv)