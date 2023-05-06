import requests
import csv

respone = requests.get("https://people.sc.fsu.edu/~jburkardt/data/csv/oscar_age_female.csv")
open('S:\III\Sem II\AI\AI_Labs_Python\Lab3\Scripts\oscar_age_female.csv', 'wb').write(respone.content)

with open('S:\III\Sem II\AI\AI_Labs_Python\Lab3\Scripts\oscar_age_female.csv') as csvfile:
    mpg = list(csv.DictReader(csvfile))
    print(mpg[:3])
