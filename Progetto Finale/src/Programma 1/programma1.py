import sys
import nltk

def main(path1, path2):
    with open(path1, "r") as corpus1:
        print(corpus1.readline())
    return

if __name__ == '__main__':
    #safe exit se non sono sono stati passati i due corpus
    if len(sys.argv)<3:
        print("Attenzione! Mancanti i corpus in input.")
    else:
        main(sys.argv[1],sys.argv[2])
    
    