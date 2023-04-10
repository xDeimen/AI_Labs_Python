import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Incarcam datele din fisierul CSV
df = pd.read_csv('S:\III\Sem II\AI\AI_Labs_Python\Lab5\case_preturi.csv')

# Alegem coloanele de intrare și ieșire
X = df[['condition', 'grade', 'yr_built', 'floors', 'sqft_living']]
y = df['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Aplicam regresia liniara pe datele de antrenare
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Facem predictii pe datele de testare
y_pred = regressor.predict(X_test)

# Calculam acuratetea modelului
accuracy = regressor.score(X_test, y_test)
print('Acuratetea modelului: ', accuracy)


# Facem o noua predictie pentru o casa cu urmatoarele caracteristici
# condition = 5, grade = 8, yr_built = 1990, floors = 2, sqft_living = 2000
new_data = [[5, 8, 1990, 2, 2000]]
predicted_price = regressor.predict(new_data)

print('Pretul prezis pentru casa data este: ', predicted_price)
