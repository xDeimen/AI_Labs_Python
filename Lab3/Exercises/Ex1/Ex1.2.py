from csv import writer

"""
METODA ASTA STERGE TOT DIN CSV
"""


lista_date = ['1', 'Stelian', 'Brad']
director = "S:\III\Sem II\AI\AI_Labs_Python\Lab3\Exercises\Ex1"
fiser = "\scriere.csv"
cale = director + fiser

with open(cale, 'w') as f:
    w = writer(f)
    w.writerow(lista_date)
    f.close()