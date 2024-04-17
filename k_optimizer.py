import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

model_path = "Models"
fig_path = "Figures"
n_components = 100

with open(f"{model_path}/extracted_features.pickle", "rb") as f:
    data = pickle.load(f)
filenames = np.array(list(data.keys()))
features = np.array(list(data.values())).reshape(-1, 4096)

# Reduce the amount of dimensions in the feature vector
print("Performing PCA...")
pca = PCA(n_components=n_components, random_state=42)
pca.fit(features)
x = pca.transform(features)

# determine optimal value for k
sse = []
list_k = list(range(3, 100))
print(f"Performing K-Means for {list_k[0]} to {list_k[-1]} clusters...")
for k in list_k:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(x)

    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse)
plt.xlabel(r"Number of clusters *k*")
plt.ylabel("Sum of squared distance")
plt.savefig(f"{fig_path}/elbow_curve")
