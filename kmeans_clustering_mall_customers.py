
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load the dataset
df = pd.read_csv('Mall_Customers.csv')
print("First 5 rows of the dataset:")
print(df.head())

# Drop non-numeric columns
data = df.drop(['CustomerID', 'Gender'], axis=1)

# Elbow Method to find optimal K
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.savefig('elbow_method.png')
plt.show()

# Apply K-Means with optimal K (e.g., 5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(data)

# Add cluster labels to data
data['Cluster'] = labels

# PCA for 2D visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data.drop('Cluster', axis=1))

# Plot the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=labels, palette='Set1')
plt.title('K-Means Clustering Visualization')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Cluster')
plt.savefig('clusters_visualization.png')
plt.show()

# Silhouette Score
score = silhouette_score(data.drop('Cluster', axis=1), labels)
print(f"Silhouette Score: {score:.2f}")
