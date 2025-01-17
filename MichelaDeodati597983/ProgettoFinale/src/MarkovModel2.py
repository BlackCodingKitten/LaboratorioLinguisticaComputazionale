import random
from collections import defaultdict
import nltk 

class MarkovModel2:
    def __init__(self,tokens):
        self.modello = {} #dizionario per memorizzare il livello
        self.build_model(tokens) #costruisce il modello a partire dalla lista di tokens
    
    def build_model(self, tokens):
        for i in range(len(tokens)-2):
            stato = (tokens[i],tokens[i+1])
            if stato not in self.modello:
                self.modello[stato] = []
            self.modello[stato].append(tokens[i+2])
            
    def getProbability(self, sentence):
        tokens = nltk.tokenize.word_tokenize(sentence)
        prob = 1
        for i in range(len(tokens)-2):
            stato_corrente = (tokens[i], tokens[i+1])
            next_word = tokens[i+2]
            if stato_corrente in self.modello:
                #conta quante volte appare next_word come prossima parola
                nw_counter = self.modello[stato_corrente].count(next_word)
                state_counter = len(self.modello[stato_corrente])
                prob *= nw_counter / state_counter
            else:
                return 0
        return prob
                