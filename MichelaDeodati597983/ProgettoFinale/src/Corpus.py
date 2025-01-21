from collections import Counter
import nltk

class Corpus:
    """
    Classe che rappresenta un corpus di testo. Fornisce metodi per gestire e manipolare
    diverse proprietà del corpus, come il testo originale, i token, il vocabolario,
    le frasi e i lemmi.

    Args:
        filePath (str): Il percorso del file contenente il corpus.

    Attributi:
        text (str): Il contenuto del testo del corpus.
        tokenList (list of str): Lista dei token estratti dal testo.
        v (dict): Vocabolario del corpus, con token come chiavi e frequenze come valori.
        sentence (list of str): Lista delle frasi estratte dal testo.
        lemmaList (list of str): Lista dei lemmi estratti dal testo.
        filePath (str): Percorso del file del corpus.
        fileName (str): Nome del file estratto dal percorso.
    """

    def __init__(self, filePath, text, sortFlag):
        """
        Inizializza un'istanza della classe Corpus con il percorso del file e inizializza
        tutti gli attributi a valori predefiniti vuoti.
        """
        self.text = text  # Contenuto testuale del corpus
        self.tokenList = nltk.tokenize.word_tokenize(text)  # Lista di token (parole e simboli) estratti
        if sortFlag:
            self.tokenList = sorted(self.tokenList)
        self.v = dict(Counter(self.tokenList)) # Vocabolario del corpus con frequenze dei token
        self.sentence = nltk.tokenize.sent_tokenize(text)  # Lista di frasi estratte dal testo
        self.lemmaList = []  # Lista di lemmi estratti dal testo
        self.filePath = filePath  # Percorso completo del file
        # Estraggo il nome del file dal percorso
        self.fileName = filePath.split("/")[-1]

    # ----------------
    # Metodi SETTER
    # ----------------

    def setText(self, text):
        """
        Imposta il contenuto testuale del corpus.

        Args:
            text (str): Testo da assegnare al corpus.

        Returns:
            bool: True se l'operazione è completata.
        """
        if type(text) != str:
            return False
        self.text = text
        return True

    def setTokenList(self, tokenList):
        """
        Imposta la lista di token del corpus.

        Args:
            tokenList (list of str): Lista di token.

        Returns:
            bool: True se l'operazione è completata.
        """
        if type(tokenList) != list:
            return False
        for t in tokenList:
            if type(t) != str:
                return False 
        self.tokenList = tokenList
        return True

    def setVocabulary(self, v):
        """
        Calcola e imposta il vocabolario del corpus utilizzando collections.Counter.
        Il vocabolario è un dizionario con token unici come chiavi e le loro frequenze come valori.

        Returns:
            bool: True se il vocabolario è stato impostato correttamente,

        """
        # Utilizza Counter per calcolare le frequenze dei token
        if type(v) != dict:
            return False
        self.v = v
        return True


    def setLemmaList(self, lemmaList):
        """
        Imposta la lista di lemmi del corpus.

        Args:
            lemmaList (list of str): Lista di lemmi.

        Returns:
            bool: True se l'operazione è completata.
        """
        # if type(lemmaList) != list:
        #     return False
        self.lemmaList = lemmaList
        return True

    def setSentenceList(self, lista):
        """
        Imposta la lista di frasi del corpus.

        Args:
            lista (list of str): Lista di frasi.

        Returns:
            bool: True se l'operazione è completata.
        """
        if type(lista) != list:
            return False
        for s in lista:
            if type(s) != str:
                return False
        self.sentence = lista
        return True

    # ----------------
    # Metodi GETTER
    # ----------------

    def getTokenList(self):
        """
        Restituisce la lista di token del corpus.

        Returns:
            list of str: Lista di token.
        """
        return self.tokenList

    def getFileName(self):
        """
        Restituisce il nome del file del corpus.

        Returns:
            str: Nome del file.
        """
        return self.fileName

    def getText(self):
        """
        Restituisce il contenuto testuale del corpus.

        Returns:
            str: Testo del corpus.
        """
        return self.text

    def getVocabulary(self):
        """
        Restituisce il vocabolario del corpus.

        Returns:
            dict: Dizionario con i token unici come chiavi e le loro frequenze come valori.
        """
        return self.v

    def getLemmaList(self):
        """
        Restituisce la lista di lemmi del corpus.

        Returns:
            list of str: Lista di lemmi.
        """
        return self.lemmaList

    def getSentenceList(self):
        """
        Restituisce la lista di frasi del corpus.

        Returns:
            list of str: Lista di frasi.
        """
        return self.sentence
