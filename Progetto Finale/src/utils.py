import nltk

def readFile(filePath):
    try:
        filePtr = open(filePath, "r")
        fileContent = filePtr.read()
        filePtr.close()
        return fileContent
    except FileNotFoundError:
        print("Errore il file "+ filePath+" non esite.")
    except IOError :
        print("Errore durante la lettura del file")
    except PermissionError:
        print("Errore non hai i permessi per aprire questo file.")

def sentenceSplitter (text):
    sencenceList = nltk.tokenize.sent_tokenize(text)
    return sencenceList

def tokenSplitter (text):
    tokenList = nltk.tokenize.word_tokenize(text)
    # ordino alfabeticamente la lista dei tokens
    return sorted(tokenList)

def writeFile(filePath, toWrite): 
    with open(filePath, "w") as filePtr:
        filePtr.write(toWrite)
        filePtr.close()