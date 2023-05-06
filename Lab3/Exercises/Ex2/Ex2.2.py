from csv import writer
from csv import reader


director = "S:\III\Sem II\AI\AI_Labs_Python\Lab3\Exercises\Ex2"
fisier_1 = "\citire.csv"
fisier_2 = "\scriere.csv"

def add_column_in_csv(cale_c, cale_s, transform_row):
    with open(cale_c, 'r') as citire, open(cale_s, 'w', newline="") as scriere:
        csv_citire = reader(citire)
        csv_scriere = writer(scriere)

        for linie in csv_citire:
            transform_row(linie, csv_citire.line_num)
            csv_scriere.writerow(linie)

cale_1 = director + fisier_1
cale_2 = director + fisier_2
default_text = "Cluj"
add_column_in_csv(cale_1,cale_2, lambda linie, line_num: linie.append(default_text))
