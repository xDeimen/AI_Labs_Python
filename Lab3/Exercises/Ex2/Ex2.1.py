from csv import writer
from csv import reader

default_text = 'Observatii'
director = "S:\III\Sem II\AI\AI_Labs_Python\Lab3\Exercises\Ex2"
fisier_1 = "\citire.csv"
fisier_2 = "\scriere.csv"

cale_1 = director + fisier_1
cale_2 = director + fisier_2

with open(cale_1, 'r') as citire, open(cale_2,'w',newline="") as scriere:
    csv_citire = reader(citire)
    csv_scriere = writer(scriere)

    for linie in csv_citire:
        linie.append(default_text)
        csv_scriere.writerow(linie)
