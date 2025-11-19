# Implement-K-MEANS-Clustring-Algorithm
python program to implement  K-MEANS Clustring Algorithm

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Step 2: Create a sample dataset
data = pd.DataFrame({
    'Annual Income (k$)': [15, 16, 17, 18, 20, 25, 28, 30, 35, 40, 42, 45, 48, 50, 55, 60, 65, 70, 72, 75],
    'Spending Score (1-100)': [39, 81, 6, 77, 40, 76, 94, 72, 36, 65, 50, 59, 60, 62, 61, 63, 48, 52, 42, 49]
})
print("----- Dataset -----")
print(data, "\n")
# Step 3: Feature selection
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
# Step 4: Normalize features (optional but helps K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Step 5: Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
# Step 6: Get cluster labels and centroids
data['Cluster'] = kmeans.labels_
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
print("----- Cluster Assignments -----")
print(data, "\n")
print("----- Cluster Centers (original scale) -----")
print(pd.DataFrame(centroids, columns=X.columns), "\n")
# Step 7: Visualize clusters
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for i in range(3):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'],
                c=colors[i], label=f'Cluster {i}')

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='yellow', marker='*', label='Centroids')

plt.title('K-Means Clustering (3 Clusters)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()
