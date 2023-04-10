import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Incarcam datele din fisiinearerul CSV
df = pd.read_csv('S:\III\Sem II\AI\AI_Labs_Python\Lab5\salarii_tabel.csv')

# Alegem coloana de intrare si iesire
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Transformam datele de intrare in forma polinomiala
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

# Aplicam regresia liniara pe datele polinomiale
regressor = LinearRegression()
regressor.fit(X_poly, y)

# Facem predictii pe datele de intrare
y_pred = regressor.predict(poly_reg.fit_transform(X))

# Desenam graficul de dispersie si curba de regresie
plt.scatter(X, y, color='red')
plt.plot(X, y_pred, color='blue')
plt.title('Regresia polinomiala')
plt.xlabel('Nivel de experienta')
plt.ylabel('Salariu')
plt.show()


# Facem o noua predictie pentru un nivel de experienta de 6.5 ani
new_data = [[6.5]]
predicted_salary = regressor.predict(poly_reg.transform(new_data))

print('Salariul prezis pentru un nivel de experienta de 6.5 ani este: ', predicted_salary)

