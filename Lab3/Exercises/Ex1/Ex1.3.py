from csv import writer

"""
METODA ASTA MERGE CA SI 1.1 DOAR CA INTEREAZA
"""


lista_date = ['1', 'Stelian', 'Brad']
director = "S:\III\Sem II\AI\AI_Labs_Python\Lab3\Exercises\Ex1"
fiser = "\scriere.csv"
cale = director + fiser

with open(cale, 'a', newline="") as f:
    for i in range(0, 4):
        w = writer(f)
        w.writerow(lista_date)
    f.close()
