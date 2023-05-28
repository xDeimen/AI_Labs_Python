import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load and analyze the dataset
def convert_plus_to_2(value):
    if value == '+':
        return 2
    return float(value)

# Load the dataset, specify data types and handle missing values
df = pd.read_csv(r"S:\III\Sem II\AI\AI_Labs_Python\Lab7\moscow_real_estate_sale.csv", dtype={'rooms': object, 'total_area': float}, na_values='?')

# Drop columns with mixed types
df = df.select_dtypes(exclude=['object'])

if 'rooms' in df.columns:
    df['rooms'] = df['rooms'].apply(convert_plus_to_2)

# Drop rows with missing values
df = df.dropna()

# Perform data analysis, handle anomalies, asymmetries, and correlated columns
# Select relevant input columns (features) and the target variable (price)
X = df[['minutes', 'storey', 'storeys', 'total_area']]  # Update with relevant columns
y = df['price']

# Step 2: Implement gradient descent algorithm

def cost_function(X, y, theta):
    # Calculate the cost function based on Mean Squared Error (MSE)
    m = len(y)
    predictions = np.dot(X, theta)
    error = predictions - y
    cost = np.sum(error ** 2) / (2 * m)
    return cost

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    # Perform gradient descent to update model parameters
    m = len(y)
    cost_history = []

    for iteration in range(num_iterations):
        predictions = np.dot(X, theta)
        error = predictions - y
        gradient = np.dot(X.T, error) / m
        theta -= learning_rate * gradient
        cost = cost_function(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

# Choose initial values for model parameters
theta = np.zeros(X.shape[1])  # Initialize with zeros
learning_rate = 0.01
num_iterations = 1000

# Run gradient descent
theta, cost_history = gradient_descent(X, y, theta, learning_rate, num_iterations)

# Step 3: Evaluate the model

# Plot the cost history to visualize convergence
plt.plot(range(1, num_iterations + 1), cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost History')
plt.show()

# Make predictions using the trained model
y_pred = np.dot(X, theta)

# Calculate evaluation metrics
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)
