import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv(r"S:\III\Sem II\AI\AI_Labs_Python\Lab7\Crypto_currency.xlsx")

# Calculate the target variable
df["Value"] = df["Volume"] * df["Close"]

# Check for missing values in the target variable
imputer_y = SimpleImputer()
y = imputer_y.fit_transform(df[["Value"]])

# Prepare the features and target variables
X = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values in features with SimpleImputer
imputer_X = SimpleImputer()
X_train = imputer_X.fit_transform(X_train)
X_test = imputer_X.transform(X_test)

# Perform K-NN regression
k_values = range(1, 21)
mse_values = []

for k in k_values:
    # Create and fit the KNeighborsRegressor model
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    
    # Predict the target variable
    y_pred = model.predict(X_test)
    
    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    mse_values.append(mse)

# Find the optimal number of k neighbors
optimal_k = k_values[mse_values.index(min(mse_values))]

print("Optimal number of k neighbors:", optimal_k)

