Step 1: Load and Visualize Dataset 

from google.colab import files
uploaded = files.upload()

import pandas as pd
df = pd.read_csv('Mall_Customers.csv')

print(df.head())

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]  # use more features if desired
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA View of Customers")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

Output:

CustomerID  Gender  Age  Annual Income (k$)  Spending Score (1-100)
0           1    Male   19                  15                      39
1           2    Male   21                  15                      81
2           3  Female   20                  16                       6
3           4  Female   23                  16                      77
4           5  Female   31                  17                      40

![image](https://github.com/user-attachments/assets/71775d20-d6b8-41c3-979a-3a24b7409ad2)

Step 2: Fit K-Means and Assign Cluster Labels

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

Step 3: Use the Elbow Method to Find Optimal K

inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

Output:
![image](https://github.com/user-attachments/assets/1edaf4b9-3b90-4e34-9557-3e7320cbfd5a)

 Step 4: Visualize Clusters with Color-Coding

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)
colors = ['red', 'blue', 'green', 'purple', 'orange']
plt.figure(figsize=(8, 6))
for i in range(5):
    plt.scatter(X[df['Cluster'] == i]['Annual Income (k$)'],
                X[df['Cluster'] == i]['Spending Score (1-100)'],
                s=50, c=colors[i], label=f'Cluster {i}')
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='yellow', marker='X', label='Centroids')

plt.title('Customer Segments via K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()

Output:
![image](https://github.com/user-attachments/assets/7abfb9c7-92df-4849-b457-ac1236263b8a)

Step 5: Evaluate Clustering Using Silhouette Score

from sklearn.metrics import silhouette_score
score = silhouette_score(X, df['Cluster'])
print(f'Silhouette Score: {score:.3f}')

Output:
Silhouette Score: 0.554













