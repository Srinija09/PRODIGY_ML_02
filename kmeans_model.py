import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset (update path after downloading from Kaggle)
data = pd.read_csv('Mall_Customers.csv')

# Select relevant features (example: Annual Income & Spending Score)
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Apply K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Print cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Visualization
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=data['Cluster'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation')
plt.show()
