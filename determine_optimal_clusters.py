import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from keras.applications import VGG16
from keras.models import Model

from main import images

images_rgb = np.repeat(images, 3, axis=-1)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

features = model.predict(images_rgb)
features_flat = features.reshape((features.shape[0], -1))

wss = []
max_clusters = 20
for i in range(1, max_clusters+1):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(features_flat)
    wss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, max_clusters+1), wss, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WSS (Within-Cluster Sum of Squares)')
plt.show()

silhouette_scores = []
for i in range(2, max_clusters+1):
    kmeans = KMeans(n_clusters=i, random_state=42)
    clusters = kmeans.fit_predict(features_flat)
    silhouette_avg = silhouette_score(features_flat, clusters)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(10, 6))
plt.plot(range(2, max_clusters+1), silhouette_scores, marker='o')
plt.title('Silhouette Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()