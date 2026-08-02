# Hey! This is my own basic implementation of the KMeans clustering algorithm.
# I wrote this to understand how the centroids actually shift step-by-step.
# (Hope it's clean enough to follow!)

import random
import numpy as np

class KMeans:
    def __init__(self, n_clusters=2, max_iter=100):
        # Setting up parameters. Defaulting to 2 clusters and 100 iterations.
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self, X):
        # Step 1: Pick random data points to act as our starting centroids.
        # X.shape[0] gives the total number of rows.
        random_index = random.sample(range(0, X.shape[0]), self.n_clusters)
        self.centroids = X[random_index]

        # Step 2: Loop iteration starts. We either run max_iter times or stop early if we converge.
        for i in range(self.max_iter):
            # assign each point to the closest centroid
            cluster_group = self.assign_clusters(X)
            
            # Save the old centroids so we can compare and check if they moved
            old_centroids = self.centroids
            
            # recalculate and move centroids to the mean of their cluster points
            self.centroids = self.move_centroids(X, cluster_group)
            
            # Converged? (i.e. did the centroids stop moving completely?)
            if (old_centroids == self.centroids).all():
                # print(f"Stopped early at iteration {i} because centroids converged!")
                break

        return cluster_group

    def assign_clusters(self, X):
        cluster_group = []
        distances = []

        # Find distance of every data point from each centroid
        for row in X:
            for centroid in self.centroids:
                # Calculating Euclidean distance using dot product (sqrt of dot product of difference vector)
                distances.append(np.sqrt(np.dot(row - centroid, row - centroid)))
            
            # Find the index of the minimum distance - that's our cluster ID!
            min_distance = min(distances)
            index_pos = distances.index(min_distance)
            
            cluster_group.append(index_pos)
            distances.clear()  # Empty the list for the next row/data point

        return np.array(cluster_group)

    def move_centroids(self, X, cluster_group):
        new_centroids = []

        # Find the unique cluster IDs present in the current assignment
        cluster_type = np.unique(cluster_group)

        for type in cluster_type:
            # Calculate the mean of all points assigned to this cluster type to find the new center
            new_centroids.append(X[cluster_group == type].mean(axis=0))

        return np.array(new_centroids)


