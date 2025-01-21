import nltk 


class MarkovModel2:
    """
    Classe che implementa un modello di Markov di ordine 2 basato su una lista di token. 
    Questo modello prevede la probabilità di una parola basandosi sulle due parole precedenti.
    """
    def __init__(self, tokens):
        """
        Inizializza un'istanza della classe MarkovModel2.

        Args:
            tokens (list): Lista di token da cui costruire il modello.
        """
        self.modello = {}  # Dizionario per memorizzare gli stati e le transizioni.
        self.build_model(tokens)  # Costruisce il modello a partire dalla lista di token.
    
    def build_model(self, tokens):
        """
        Costruisce il modello di Markov basato su una lista di token.

        Args:
            tokens (list): Lista di token da cui costruire il modello.
        """
        for i in range(len(tokens) - 2):
            # Crea uno stato formato da una coppia di parole consecutive.
            stato = (tokens[i], tokens[i+1])
            if stato not in self.modello:
                # Inizializza una lista vuota per lo stato se non esiste già.
                self.modello[stato] = []
            # Aggiunge la parola successiva allo stato attuale.
            self.modello[stato].append(tokens[i+2])
            
    def getProbability(self, sentence):
        """
        Calcola la probabilità di una frase basandosi sul modello di Markov.

        Args:
            sentence (str): La frase di cui calcolare la probabilità.

        Returns:
            float: La probabilità della frase nel modello.
                   Ritorna 0 se uno degli stati non è presente nel modello.
        """
        tokens = nltk.tokenize.word_tokenize(sentence)  # Tokenizza la frase.
        prob = 1  # Probabilità iniziale.
        for i in range(len(tokens) - 2):
            # Determina lo stato corrente come coppia di parole consecutive.
            stato_corrente = (tokens[i], tokens[i+1])
            next_word = tokens[i+2]  # Parola successiva nello stato corrente.
            if stato_corrente in self.modello:
                # Conta quante volte next_word appare come parola successiva.
                nw_counter = self.modello[stato_corrente].count(next_word)
                # Conta il numero totale di transizioni dallo stato corrente.
                state_counter = len(self.modello[stato_corrente])
                # Aggiorna la probabilità moltiplicando per la probabilità condizionata.
                prob *= nw_counter / state_counter
            else:
                # Se lo stato non è presente nel modello, la probabilità è zero.
                return 0
        return prob

    def getSmoothProbability(self, sentence):
        """
        Calcola la probabilità di una frase basandosi sul modello di Markov,
        utilizzando la stima di Laplace (add-one smoothing) per evitare probabilità zero.

        Args:
            sentence (str): La frase di cui calcolare la probabilità.

        Returns:
            float: La probabilità della frase nel modello.
        """
        # Tokenizza la frase in parole
        tokens = nltk.tokenize.word_tokenize(sentence)
        prob = 1  # Probabilità iniziale impostata a 1
        
        for i in range(len(tokens) - 2):
            # Determina lo stato corrente come una coppia di parole consecutive
            stato_corrente = (tokens[i], tokens[i+1])
            next_word = tokens[i+2]  # Parola successiva nello stato corrente
            
            # Controlla se lo stato corrente è presente nel modello
            if stato_corrente in self.modello:
                # Conta il numero di occorrenze della parola successiva nello stato corrente
                nw_counter = self.modello[stato_corrente].count(next_word)
                # Conta il numero totale di transizioni dallo stato corrente
                state_counter = len(self.modello[stato_corrente])
            else:
                # Se lo stato corrente non è nel modello, assume contatori a zero
                nw_counter = 0
                state_counter = 0
            
            # Calcola il numero totale di parole nel vocabolario del modello
            vocab_size = len(set(word for words in self.modello.values() for word in words))
            
            # Applica la stima di Laplace
            prob *= (nw_counter + 1) / (state_counter + vocab_size)
        
        return prob
