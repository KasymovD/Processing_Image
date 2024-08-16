import os
from PIL import Image
import numpy as np


def load_and_preprocess_images(folder, output_folder, img_size=(128, 128)):
    images = []
    labels = []
    filenames = []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        output_subfolder = os.path.join(output_folder, subfolder)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                try:
                    pil_image = Image.open(img_path).convert('L')
                    pil_image = pil_image.resize(img_size)
                    img = np.array(pil_image)
                    images.append(img)
                    labels.append(subfolder)
                    filenames.append(filename)

                    output_path = os.path.join(output_subfolder, filename)
                    pil_image.save(output_path)
                except Exception as e:
                    print(f"Warning: Could not open or process the image {img_path}. Error: {e}")
    images = np.array(images)
    images = np.expand_dims(images, axis=-1)
    return images, np.array(labels), filenames


data_folder = r'C:\Users\user\PycharmProjects\Clustering\kcgData'
output_folder = r'C:\Users\user\PycharmProjects\Clustering\processed_images'
images, labels, filenames = load_and_preprocess_images(data_folder, output_folder)
