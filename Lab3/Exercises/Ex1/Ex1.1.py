from csv import writer

"""
METODA ASTA ADAUGA GRADUAL
"""


lista_date = ['1','Huszar','Istvan']
director = "S:\III\Sem II\AI\AI_Labs_Python\Lab3\Exercises\Ex1"
fiser = "\scriere.csv"
cale = director + fiser

with open(cale, 'a', newline="") as f:
    w = writer(f)
    w.writerow(lista_date)
    f.close()
