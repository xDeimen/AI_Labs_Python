from csv import DictWriter

"""
Scriem dictionar
Afiseaza valoarea cheilor din dict
"""

cap_tabel = ['NR', 'NUME', 'SUBIECT']

dictionar = {'NR':'04','NUME':'Huszar Istvan','SUBIECT':'Robotica'}
director = "S:\III\Sem II\AI\AI_Labs_Python\Lab3\Exercises\Ex1"
fiser = "\scriere.csv"
cale = director + fiser

with open(cale, 'a', newline="") as f:
    dw = DictWriter(f, fieldnames=cap_tabel)
    dw.writerow(dictionar)

f.close()