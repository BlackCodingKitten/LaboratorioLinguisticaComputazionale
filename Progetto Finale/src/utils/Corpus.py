



class Corpus:
    def __init__(self, tokens, path):
        #
        self.nome_corpus = ((path.split("/"))[-1])
        self.lista_token = tokens
        #costruisco il vocabolaraio dei tipi come un dizionario con chiave le parole tipo e valore la loro frequenza assoluta, il vocabolario è ordinato
        self.vocabolario = dict([(x,(self.lista_token).count(x)) for x in sorted(set(self.lista_token))])
        #da ampliare
    
        
    def getTokens(self):
        """restituisce la lista dei token nel corpus

        Returns:
            list:lista di stringhe token 
        """        
        return self.lista_token
    
    def getVocabolario(self):
        """restituisce il vocabolario dei tipi con relative frequenze

        Returns:
            dict:izionario con chiave le parole tipo e valore la loro frequenza assoluta nel corpus
        """   
        return self.vocabolario
    
    def V(self):
        """funzione che restituicse la lunghezza del vocabolario dei tipi del corpus
        Returns:
            int: numero di parole tipo
        """    
        return len(self.vocabolario.keys()) 
    
    def C(self):
        """funzione che restituicse la lunghezza della lista dei token di un corpus
        Returns:
            int: numero di token
        """    
        return len(self.lista_token) 
      
        
    def TTR(self):
        """calcolatore del ttr di un corpus

        Returns:
            float: valore del type token ratio
        """        
        return round((self.V()/self.C()),3)