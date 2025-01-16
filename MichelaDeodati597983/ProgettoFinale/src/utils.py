import nltk  
import datetime  
import pandas as pd

# Funzione per leggere il contenuto di un file
def readFile(filePath):
    """
    Legge il contenuto di un file di testo specificato dal percorso.

    Args:
        filePath (str): Percorso del file da leggere.

    Returns:
        str: Contenuto del file se la lettura ha successo, altrimenti None.

    Gestione errori:
        - FileNotFoundError: Sollevato se il file non esiste.
        - IOError: Sollevato in caso di errori generici durante la lettura del file.
        - PermissionError: Sollevato se mancano i permessi per accedere al file.
    """
    try:
        # Apertura del file in modalità lettura
        filePtr = open(filePath, "r")
        fileContent = filePtr.read()  # Lettura del contenuto del file
        filePtr.close()  # Chiusura del file
        return fileContent
    except FileNotFoundError:  # Gestione dell'errore se il file non esiste
        print("Errore il file " + filePath + " non esiste.")
    except IOError:  # Gestione di errori generici di I/O
        print("Errore durante la lettura del file.")
    except PermissionError:  # Gestione di errori di permesso
        print("Errore non hai i permessi per aprire questo file.")

# Funzione per dividere un testo in frasi
def sentenceSplitter(text):
    """
    Divide un testo in una lista di frasi.

    Args:
        text (str): Testo da suddividere.

    Returns:
        list: Lista di frasi estratte dal testo.
    """
    # Utilizzo del tokenizer di nltk per dividere il testo in frasi
    sencenceList = nltk.tokenize.sent_tokenize(text)
    return sencenceList

# Funzione per dividere un testo in token
def tokenSplitter(text):
    """
    Divide un testo in una lista di token e la ordina alfabeticamente.

    Args:
        text (str): Testo da suddividere in token.

    Returns:
        list: Lista ordinata alfabeticamente dei token estratti.
    """
    # Utilizzo del tokenizer di nltk per dividere il testo in parole
    tokenList = nltk.tokenize.word_tokenize(text)
    # Ordino alfabeticamente la lista dei token
    return tokenList

# Funzione per creare il percorso di un file di risultati
def crateResultsFilePath():
    """
    Genera il percorso di base per salvare i risultati in una directory 'results'.

    Returns:
        str: Percorso del file di risultati con prefisso "results1_".
    """
    return  "../results/results_programma1_"

# Funzione per scrivere contenuti in un file
def writeFile(filePath, toWrite):
    """
    Scrive una stringa specificata in un file con un nome basato sulla data corrente.

    Args:
        filePath (str): Percorso di base della directory in cui salvare il file.
        toWrite (str): Contenuto da scrivere nel file.

    Effetti:
        - Salva il contenuto nel file "results1_<data_corrente>.txt".
    """
    # Apertura del file in modalità scrittura con nome basato sulla data odierna
    with open(filePath + f"{datetime.date.today()}.txt", "w") as filePtr:
        filePtr.write(toWrite)  # Scrittura del contenuto nel file
        filePtr.close()  # Chiusura del file (opzionale, implicito con 'with')

def tupleListToString(lista):
    """
    Converte una lista di tuple in una lista di stringhe formattate come tuple.

    Ogni elemento della lista di input è una tupla. La funzione crea una rappresentazione
    in formato stringa di ciascuna tupla e restituisce una nuova lista contenente queste stringhe.

    Args:
        lista (list): Una lista di tuple. Ogni tupla può contenere uno o più elementi.

    Returns:
        list: Una lista di stringhe, dove ogni stringa rappresenta una tupla formattata
              con elementi separati da virgole e racchiusi tra parentesi tonde.
    """
    # Inizializza una lista vuota che conterrà le stringhe da restituire
    toReturn = []
    
    # Itera su ogni elemento della lista (ciascuno deve essere una tupla)
    for e in lista:
        # Inizia la rappresentazione stringa con il primo elemento della tupla
        s = f"({e[0]}"
        
        # Itera sugli elementi successivi della tupla (dal secondo in poi)
        for i in range(1, len(e)):
            # Aggiunge ogni elemento alla stringa, preceduto da una virgola e uno spazio
            s += f", {e[i]}"
        
        # Chiude la rappresentazione della tupla con una parentesi tonda
        s += ")"
        
        # Aggiunge la stringa completa alla lista di output
        toReturn.append(s)
    
    # Restituisce la lista di stringhe formattate
    return toReturn



# funzione che formatta l'output
def createTable(tValue, dfColumns, dfIndex, indexFlag = True):
    """funzione che crea un dataframe pandas con i valori passati e li formatta in una stringa

    Args:
        tValue (list): lista dei valori da inserire nella tabella 
        dfColumns (list): nome delle colonne da inserire
        dfIndex (list): indicizzazione per rirga della tabella
    """    
     #creo un dataframe in pandas per formattare l'output
    dataFrame = pd.DataFrame(tValue, columns=dfColumns, index=dfIndex)
    
    return "\n"+dataFrame.to_string(index = indexFlag)+"\n"