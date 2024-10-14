import datetime

def year2(age=int(input("Quanti anni hai?"))):
    currY=datetime.datetime.now().year
    hundredY=100-age
    return currY + hundredY
    

def fun (a,b):
    if a*b <= 1000:
        return a*b
    return a+b


def year():
    a=input("Immetti la data di nascita  nel formato GG/MM/AAAA ")
    l=a.split("/")
    l[2]=str(int(l[2])+100)
    return "/".join(l)
 
# a=int(input("Immetti a: "))
# b=int(input("Immetti b: "))

# print("",fun(a,b))

# print("Il risultato calcolato e'",year())
print("Compirai 100 anni nel: ", year2())