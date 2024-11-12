""" Modello di markov di ordine 1 -> bigrammi 
Modello di ordine 2 -> trigrammi ecc
Modello di ordine 0-> bag of words 
"""
import nltk
def markov0 (frase, freq_distr, C):
    tokens = nltk.tokenize.word_tokenize(frase)
    prob = 1
    for token in tokens:
        token_prob = freq_distr[token]/C #prblema nel chainign se ho almeno un termine che è zero tutto il prodotto è zero
        prob*=token_prob
    return prob



#RICORDATI DI USARE FREQ DIST CON NLTK  PRE LE DISTRIBUZIONI DI FREQUENZE

        