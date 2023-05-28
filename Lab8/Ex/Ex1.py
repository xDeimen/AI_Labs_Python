import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv(r"S:\III\Sem II\AI\AI_Labs_Python\Lab8\case.csv")

# Separate the categorical column(s)
categorical_cols = ['proximitate_statiune_turistica']
numeric_cols = [col for col in df.columns if col not in categorical_cols]

# Perform feature scaling on the numeric columns
scaler = StandardScaler()
imputer = SimpleImputer(strategy='median')

# Fill missing values in numeric columns
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
scaled_features = scaler.fit_transform(df[numeric_cols])

# Perform one-hot encoding on the categorical column(s)
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(df[categorical_cols])

# Combine the scaled numeric features and encoded categorical features
all_features = pd.DataFrame(data=scaled_features, columns=numeric_cols)
encoded_columns = encoder.get_feature_names_out(categorical_cols)
all_features[encoded_columns] = encoded_features

# Determine the optimal value for k
inertia = []
k_values = range(1, 11)

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(all_features)
    inertia.append(model.inertia_)

# Plot the elbow curve to find the optimal k
plt.plot(k_values, inertia, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Curve')
plt.show()

# Determine the optimal value of k based on the elbow point
optimal_k = int(input("Enter the optimal value of k: "))

# Part 2: Modifying the "cases.csv" file to "houses.csv" and replacing values in the "Location" column

# Create a copy of cases.csv and name it houses.csv
df_copy = df.copy()
df_copy.to_csv("houses.csv", index=False)

# Replace the name of the last column with 'Rent'
df_copy.rename(columns={'Unnamed: 2': 'Rent'}, inplace=True)

# Find the count of specific values in the 'Location' column
location_counts = df_copy['Location'].value_counts()

# Replace values in the 'Location' column with their English equivalents
mapping = {
    'IN STATION': 'In the resort',
    'APPROX. 1 HOUR DISTANCE': '< 1 h distance',
    'NEARBY': 'in proximity',
    'NEAR THE RESORT': 'nearby',
    'FAR': 'far distance'
}
df_copy['Location'] = df_copy['Location'].map(mapping)

# Print the updated count of values in the 'Location' column
print(location_counts)
