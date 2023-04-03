import pandas as pd

data = pd.read_csv("S:\\III\\Sem II\\AI\\AI_Labs_Python\\Lab4\\nba.csv", index_col="N")
print(data)
first = data.loc["Mike_Hawk"]
second = data.loc["Avery_Bradley"]

print(first, "\n\n\n", second)
#folosim metoda Lo
