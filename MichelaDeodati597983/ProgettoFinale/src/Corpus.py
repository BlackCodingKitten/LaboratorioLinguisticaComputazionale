
class Corpus:
    
    def __init__(self, filePath):
         # salvo il path del file 
        self.text = ""
        self.tokenList = []
        self.v = {}
        self.sentence = []
        self.lemmaList = []
        # estraggo il nome del file
        self.filePath = filePath
        self.fileName = (filePath.split("/")[-1])
        
       
    
    # Metodi SETTER 
    def setText(self, text):
        self.text = text
        return True
    
    def setTokenList(self, tokenList):
        self.tokenList = tokenList
        return True

    def setVocabulary(self):
        if len(self.tokenList) == 0:
            return False
        for t in sorted(set((self.tokenList))):
            self.v[t] = self.tokenList.count(t)
        return True
    
    def setLemmaList(self, lemmaList):
        self.lemmaList = lemmaList
        return True
    
    def setSentenceList(self, lista):
        self.sentence = lista
        return True
    
    # Metodi GETTER
    def getTokenList(self):
        return self.tokenList
    
    def getFileName(self):
        return self.fileName

    def getText(self):
        return self.text
    
    def getVocabulary(self):
        return self.v
    
    def getLemmaList(self):
        return self.lemmaList
    
    def getSentenceList(self):
        return self.sentence
    
    
    