import numpy as np
from sklearn.metrics import pairwise_distances
from clustering import features_flat, clusters, num_clusters, kmeans, filenames
import matplotlib.pyplot as plt

distances = pairwise_distances(features_flat, kmeans.cluster_centers_)

plt.figure(figsize=(15, 10))

for i in range(num_clusters):
    cluster_indices = np.where(clusters == i)[0]
    cluster_distances = distances[cluster_indices, i]

    mean_distance = np.mean(cluster_distances)
    min_distance = np.min(cluster_distances)
    max_distance = np.max(cluster_distances)

    plt.subplot(3, 4, i + 1)
    plt.hist(cluster_distances, bins=15, color='blue', alpha=0.7)
    plt.axvline(mean_distance, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_distance:.2f}')
    plt.axvline(min_distance, color='green', linestyle='dashed', linewidth=2, label=f'Min: {min_distance:.2f}')
    plt.axvline(max_distance, color='orange', linestyle='dashed', linewidth=2, label=f'Max: {max_distance:.2f}')

    plt.title(f'Cluster {i}')
    plt.xlabel('Distance to Centroid')
    plt.ylabel('Frequency')
    plt.legend()

plt.suptitle('Distance Distribution to Centroids per Cluster')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
