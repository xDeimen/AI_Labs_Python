import csv

director = "S:\III\Sem II\AI\AI_Labs_Python\Lab3"
fisier = "\data_banknote.csv"
cale = director + fisier

#deschidem fisierul

file = open(cale)
print(file)
print(type(file))

#citim fisierul
cititor = csv.reader(file)
print(cititor, "\n")

#afisam cap de tabel
header = []
header = next(cititor)
print(header, "\n")

#extragem date din fisier si le salvam intr-o lista
linii = []
i = 0

for x in cititor:
    linii.append(x)
    i+=1
    if i <=3:
        print(linii)

print(i)

#diverse operatuu cu date din lista

print(len(linii))
print(linii[0])
print(linii[1])
print(linii[1][3])

#inchidem fisierul
file.close()

#alta abordare

k = []
with open(cale, 'r') as fisier:
    f = csv.reader(fisier)
    h = next(f)

    for t in f:
        k.append(t)

print(h)
print(k[2])

#o alta aboradre

with open(cale) as p:
    c =p.readlines()
cap = c[:1]
primele_2_linii = c[1:3]
print(cap)
print(primele_2_linii)

