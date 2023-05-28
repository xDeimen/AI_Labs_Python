import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data from "evaluation_cars.csv"
col_names = ["pret", "maintenance cost", "number of doors", "maximum number of passengers", "luggage size", "degree of safety", "decision"]
data = pd.read_csv(r"C:\\Users\\Cristi\\Downloads\\evaluare_masini.csv", header=0, names= col_names)

#dic=["low": 2, "med": 3, "high": 4, "vhigh": 5]
# Print the column names to check if they are being read correctly
print(data.columns)
#data['pret']=data['pret'].astype(dic)

print(data.head())

# Split the data into input features (X) and output labels (y)
#X = data_encoded.drop("decision", axis=1)
#y = data_encoded["decision"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier model
clf = DecisionTreeClassifier()

# Fit the model to the training data
clf.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Preprocess the new data
new_data = pd.DataFrame([["vhigh", "med", "5more", "4", "med", "med"]], columns=X.columns)
new_data_encoded = pd.get_dummies(new_data, columns=["pret", "maintenance cost", "luggage size"])
new_data_encoded["number of doors"] = new_data_encoded["number of doors"].map({"2": 2, "3": 3, "4": 4, "5more": 5})
new_data_encoded["maximum number of passengers"] = new_data_encoded["maximum number of passengers"].map({"2": 2, "4": 4, "more": 5})

# Use the trained model to predict the label for the new data
new_data_pred = clf.predict(new_data_encoded)

print("Predicted label for new data:", new_data_pred)
