
class Corpus:
    def __init__(self, xfilePath):
        # estraggo il nome del file
        if "/" in xfilePath:
            fileName = (xfilePath.split("/")[-1])
        elif "\\" in filePath:
            fileName = (xfilePath.split("\\")[-1])
        # salvo il path del file 
        filePath = xfilePath
        text = ""
        tokenList = []
        vocabulary = {}
        sentence = []
        lemmaList = []
    
    # Metodi SETTER 
    def setText(self, text):
        self.text = text
        return True
    
    def setTokenList(self, tokenList):
        self.tokenList = tokenList
        return True

    def setVocabulary(self):
        if len(self.tokenList == 0):
            return False
        for t in set(self.tokenList):
            self.setVocabulary[t] = (self.tokenList).count(t)
        return True
    
    def setLemmaList(self, lemmaList):
        self.setLemmaList = lemmaList
        return True
    
    def setSentenceList(self, lista):
        self.sentence = lista
        return True
    
    # Metodi GETTER
    def getTokenList(self):
        return self.tokenList
    
    def getFileName (self):
        return self.fileName

    def getText(self):
        return self.text
    
    def getVocabulary(self):
        return self.vocabulary
    
    def getLemmaList(self):
        return self.lemmaList
    
    def getSentenceList(self):
        return self.sentence
    
    
    