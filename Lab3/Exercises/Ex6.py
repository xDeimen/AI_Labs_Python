import csv

director = "S:\III\Sem II\AI\AI_Labs_Python\Lab3"
fisier = "\data_banknote.csv"
cale = director + fisier

#deschidem fisierul
print("Open the file and display the file type: ")
file = open(cale)
print(type(file))
print()
with open(cale) as csvfile:
    mpg = list(csv.DictReader(csvfile))
print("Read the file in Python. Display the data from the file on the screen: ")
print(mpg)
print()

#citim fisierul
cititor = csv.reader(file)

#cap de tabel
header = []
header = next(cititor)
print("Display the content of the table header in the file:")
print(header)
print()

#extragem date din fisier si le salvam intr-o lista
linii = []
nr_linii = 0

print("Create a list containing as elements the lines from the file:")
for x in cititor:
    linii.append(x)
    nr_linii+=1
    if nr_linii <=3:
        print(linii)
print()

print("Identify the number of lines in the file and display it on the screen")
print(nr_linii)
print()
#diverse operatuu cu date din lista

print("Display the first and second lines of the file")
print(linii[0])
print(linii[1])
print()

print("Check what type and how many elements each element from the list above contains")
l=1
for linie in linii:
    e=0
    for element in linie:
        e = e + 1
        print(type(element))
    print("Number of elements on line ", l, "is: ", e)

print()

print("Check if the third element of the second element of the list above is greater or less than the fourth element of the 200th element of the list:")

print("linii[1][2];", linii[1][2])
print("linii[199][3]", linii[198][3])

if linii[1][2] >= linii[199][3]:
    print("It is bigger")
else:
    print("It is smaller")
print()
#inchidem fisierul
file.close()


