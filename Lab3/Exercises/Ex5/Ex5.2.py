from csv import DictReader
from csv import DictWriter

director = "S:\III\Sem II\AI\AI_Labs_Python\Lab3\Exercises\Ex5"
fisier_1 = "\citire.csv"
fisier_2 = "\scriere.csv"

def add_column_in_csv_2(fisier_citire, fisier_scriere, linie_transf, coloana_tr_nume):
    with open(fisier_citire, 'r') as loc_citire, open(fisier_scriere, 'w', newline="") as loc_scriere:
        citire_dictionar = DictReader(loc_citire)
        nume_campuri = citire_dictionar.fieldnames
        coloana_tr_nume(nume_campuri)

        scriere_dictionar = DictWriter(loc_scriere, nume_campuri)
        scriere_dictionar.writeheader()
        for x in citire_dictionar:
            linie_transf(x, citire_dictionar.line_num)
            scriere_dictionar.writerow(x)

cale_1 = director + fisier_1
cale_2 = director + fisier_2

header_col_noua = 'Adresa'
default_text = 'Confidential'

add_column_in_csv_2(cale_1, cale_2,
                    lambda A, line_num: A.update({header_col_noua: default_text}),
                    lambda B: B.insert(0, header_col_noua))