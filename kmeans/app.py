# Testing our custom KMeans implementation on some real student data!
# (And also keeping the synthetic data blobs code in case we want to test again later)

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from kmeans import KMeans
import pandas as pd

# --- Synthetic Data Experiments (Commented out for now) ---
# centroids = [(-5,-5),(5,5),(-2.5,2.5),(2.5,-2.5)]
# cluster_std = [1,1,1,1]
# X,y = make_blobs(n_samples=100,cluster_std=cluster_std,centers=centroids,n_features=2,random_state=2)
# plt.scatter(X[:,0],X[:,1])
# ----------------------------------------------------------

# Load actual student clustering dataset
df = pd.read_csv('student_clustering.csv')

# Extracting all values from the columns as a numpy array for our KMeans input
X = df.iloc[:,:].values

# Let's instantiate KMeans with 4 clusters and up to 500 iterations
km = KMeans(n_clusters=4, max_iter=500)
y_means = km.fit_predict(X)

# Plot the 4 clusters with different colors to visualize them clearly!
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], color='red', label='Cluster 0')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], color='blue', label='Cluster 1')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], color='green', label='Cluster 2')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], color='yellow', label='Cluster 3')

plt.title('Student Clustering Results')
plt.legend()
plt.show()