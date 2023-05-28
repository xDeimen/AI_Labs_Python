import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv(r"S:\III\Sem II\AI\AI_Labs_Python\Lab8\date_medicale.csv")

# Separate the features
X = df

# Perform dimensionality reduction using PCA with 2 components
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X)

# Visualize the principal components in a 2D scatter plot
plt.scatter(X_2d[:, 0], X_2d[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Principal Components (2D)")
plt.show()

# Perform dimensionality reduction using PCA with 3 components
pca_3d = PCA(n_components=3)
X_3d = pca_3d.fit_transform(X)

# Visualize the principal components in a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2])
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
ax.set_title("Principal Components (3D)")
plt.show()
