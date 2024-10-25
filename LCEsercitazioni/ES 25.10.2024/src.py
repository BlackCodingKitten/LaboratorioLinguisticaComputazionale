"""
• Utilizzando Python e NLTK, scrivere un programma che:
    • stampi tutti i bigrammi diversi presenti all’interno del file. Per ogni bigramma (𝑢, 𝑣) stampi
        • La frequenza dei due token e la frequenza del bigramma
        • La probabilità condizionata
        • La probabilità congiunta
    • Identifichi il bigramma con probabilità congiunta massima
"""

import nltk
import sys
import math
import datetime

# path = "/home/mikela/Documents/LaboratorioLinguisticaComputazionale/LCEsercitazioni/ES 25.10.2024/example.txt"

def readFile (filePath):
    try:
        filePtr = open(filePath, "r")
        text = filePtr.read()
        filePtr.close()
        return  text
    except IOError or  FileNotFoundError:
        print(f"Errore nell'apertura del file {(filePath.split("/"))[-1]}")

def tokenizer(text):
    return nltk.word_tokenize(text)

def createVocabulary (tokensList):
    vocabulary = {}
    for token in tokensList:
        if token in vocabulary.keys():
            vocabulary[token] +=1
        else:
            vocabulary[token] = 1

def bigramizer (text):
    return list(nltk.bigrams(text))

def outputFileGenerator (to_print):
    with open("/home/mikela/Documents/LaboratorioLinguisticaComputazionale/LCEsercitazioni/ES 25.10.2024/result.txt", "w") as outputFile:
       outputFile.write(outputFile)
       outputFile.close() 


def main(filePath):
    return





if __name__ == '__main__':
    main(sys.argv[1])
    
