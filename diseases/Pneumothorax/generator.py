import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import os
import matplotlib.pyplot as plt



LABEL_COLUMN_NAME = "Finding Labels"
INDEX_COLUM_NAME = "Image Index"
DISEASE = "Pneumothorax"


class DataGenerator(Sequence):
    def __init__(self, data, image_dir, batch_size=32):
        self.batch_size = batch_size
        self.image_dir = image_dir
        self.data = data

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch_data = self.data.iloc[idx * self.batch_size : (idx + 1) * self.batch_size]

        batch_images = []
        batch_labels = []

        for i, row in batch_data.iterrows():
            image_path = os.path.join(self.image_dir, row[INDEX_COLUM_NAME])
            label = row[LABEL_COLUMN_NAME]

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Preprocesing danych
            image = cv2.equalizeHist(image) 

            batch_images.append(image)
            batch_labels.append(label)

        return np.array(batch_images), np.array(batch_labels)

# Przykładowe użycie generatora danych

if __name__ =="__main__":
    df = pd.read_csv('../../sample_labels.csv')
    data = df[df[LABEL_COLUMN_NAME].isin([DISEASE, "No Finding"])][["Image Index", "Finding Labels"]]

    image_dir = './images'
    batch_size = 32

    data_generator = DataGenerator(data, image_dir, batch_size)
    batch_images, batch_labels = data_generator[1]

    # Wyświetlenie kilku przykładowych obrazów z etykietami
    plt.figure(figsize=(10, 8))
    for i in range(32):
        plt.subplot(8, 4, i + 1)
        plt.imshow(batch_images[i].squeeze(), cmap='gray')
        plt.title(batch_labels[i])
        plt.axis('off')
    plt.show()

