import re #modulo delle regex
#una regex(pattern da verificare) si usa come una condizione di valutazione sulla stringa e restituisce true se è soddisfatta, false alt
# path="./Lab-1/file/new-hope.txt"

# # print(re.findall("(.*)", input("Inserisci la stringa: ")))
    
# pattern = r'Morte Nera|Luke|Forza|jedi'
# pattern2 =r'C1-P8' #le regex si scrivono tra singoli apici e vanno precedute da r


# l=[]
# with open(path, "r") as file:
#     for line in file:
#         # l = (re.findall(pattern,line))
#         # if l:
#         #     print(l)
#         string = re.sub(pattern2, "R2-D2", line)
#         if(string != line):
#             print("Prima:\t", line)
#             print("\nDopo:\t",string)
            
      
#esercizio 1
r1=r'[aA]|[eE]|[iI]|[oO]|[uU]' 
r2=r'\b[qwrtypsdfghjklzxcvbnmMNBVCXZLKJHGFDSQWRTYP]\w*'
r3=r'\w+{,|;|!|?|.|:}'
r4=r'\sta?r\w*'

r6= r'[0123]+?'   
#esercizio 2  
p1= r'\d{1,3}((,|\.| )?\d{3})*((,|\.){1}\d+)?'
p2= r' \w*(sto|sito)(\n|\r\n)'
p3= r'\.(\n|\r\n)'
p4= r'([A-Z]\.){2,}'
p5= r'((0|4|2|6|8){1}(1|3|5|7|9)?)+'
p6= r'(\w| )*[A-Z](\n|\r\n)'
p7= r'[aA]mav(o|i|a|amo|ate|ano)'
p72= r'[aA][mM][aA][vV]([oO]|[iI]|[aA]|[aA][mM][oO]|[aA][tT][eE]|[aA][nN][oO])'

#espressione regolare per le date in formato dd-mm-yyyy / mm-dd-yyyy / yyyy-mm-gg / yyyy-gg-mm: 
date_expression = r'(0?\d|1\d|2\d|3[01])[-/](0?\d|1[012])[-/]((?:[0-2]\d)?\d\d)|\2[-/]\1[-/]\3|\3[-/]\2[-/]\1|\3[-/]\1[-/]\2'