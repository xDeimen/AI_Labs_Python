import csv
with open('S:\III\Sem II\AI\AI_Labs_Python\Lab3\Scripts\oscar_age_female.csv') as csvfile:
    mpg = list(csv.DictReader(csvfile))


name = set(d[' "Year"'] for d in mpg)

test = []

for k in name:
    sum = 0
    type = 0
    i = 0

    for d in mpg:
        i += 1
        if d[' "Year"'] == k:
            sum += float(d[' "Year"'])
            type += 1+i
    test.append((k, sum/type))

test.sort(key=lambda x:x[0])
print(test)