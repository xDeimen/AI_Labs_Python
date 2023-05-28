import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


n_samples = 1000
blob_centers = ([1, 0.5], [3.5, 4], [1, 3.3], [5, 1.2], [6, 6], [2, 8]) 
data, labels = make_blobs(n_samples=n_samples,
centers=blob_centers,
cluster_std=0.5,
random_state=0)
colors = ('red', 'gray', 'green', 'orange', "blue", "magenta") 
figure, axis = plt.subplots(figsize=(10,10))
for n_class in range(len(blob_centers)):
    axis.scatter(data[labels==n_class][:, 0],
        data[labels==n_class][:, 1], 
        c=colors[n_class],
        s=30,
        label=str(n_class))
plt.show()

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate the dataset
n_samples = 1000
blob_centers = ([1, 0.5], [3.5, 4], [1, 3.3], [5, 1.2], [6, 6], [2, 8])
data, labels = make_blobs(n_samples=n_samples,
                          centers=blob_centers,
                          cluster_std=0.5,
                          random_state=0)
colors = ('red', 'gray', 'green', 'orange', 'blue', 'magenta')

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)

# Create and train the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), random_state=0)
mlp.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = mlp.predict(X_test)

# Calculate the classification accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Classification Accuracy:", accuracy)
