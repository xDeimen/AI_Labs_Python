import csv
with open('S:\III\Sem II\AI\AI_Labs_Python\Lab3\Scripts\oscar_age_female.csv') as csvfile:
    mpg = list(csv.DictReader(csvfile))


print(len(mpg))
print(mpg[0].keys())
print(mpg[4].keys())

name = set(d['Index'] for d in mpg)
print(name)

name = set(d[' "Year"'] for d in mpg)
print(name)

name = set(d[' "Name"'] for d in mpg)
print(name)