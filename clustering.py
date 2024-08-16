import os
import numpy as np
from keras.applications import VGG16
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image

from main import images, filenames

images_rgb = np.repeat(images, 3, axis=-1)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

features = model.predict(images_rgb)
features_flat = features.reshape((features.shape[0], -1))

num_clusters = 12
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(features_flat)
centroids = kmeans.cluster_centers_

output_base_folder = r'C:\Users\user\PycharmProjects\Clustering\clustered_images'
if not os.path.exists(output_base_folder):
    os.makedirs(output_base_folder)

for cluster_id in range(num_clusters):
    cluster_folder = os.path.join(output_base_folder, f'cluster_{cluster_id}')
    if not os.path.exists(cluster_folder):
        os.makedirs(cluster_folder)

    cluster_indices = np.where(clusters == cluster_id)[0]
    for idx in cluster_indices:
        img = Image.fromarray((images[idx].squeeze() * 255).astype(np.uint8))
        original_filename = filenames[idx]
        img.save(os.path.join(cluster_folder, original_filename))

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
features_tsne = tsne.fit_transform(features_flat)

plt.figure(figsize=(12, 10))
plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=clusters, cmap='tab10', s=50)
plt.title(f't-SNE Clustering of Stamps with {num_clusters} Clusters')
plt.colorbar(ticks=range(num_clusters))
plt.show()

for i in range(num_clusters):
    print(f"Cluster {i} contains {np.sum(clusters == i)} images.")
