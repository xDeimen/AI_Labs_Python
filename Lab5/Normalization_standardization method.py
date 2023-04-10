import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

DF = pandas.read_csv("S:\III\Sem II\AI\AI_Labs_Python\Lab5\cars.csv")

X = DF[['Weight', 'Volume']]
y = DF['CO2']

valori_normalizate = scale.fit_transform(X)
print(valori_normalizate)

regr = linear_model.LinearRegression()
regr.fit(valori_normalizate, y)
print()
scaled = scale.transform([[2300, 1.3]])
print(scaled)
print(scaled[0])
CO2_prev = regr.predict([scaled[0]])
print(CO2_prev)
