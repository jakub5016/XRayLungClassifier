import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import os
import matplotlib.pyplot as plt



LABEL_COLUMN_NAME = "Finding Labels"
INDEX_COLUM_NAME = "Image Index"
DISEASE = "Pneumothorax"
SIZE = 256


class DataGenerator(Sequence):
    def __init__(self, data, image_dir, batch_size=32, image_size=(SIZE, SIZE), **kwargs):
        super().__init__(**kwargs)  
        self.batch_size = batch_size
        self.image_dir = image_dir
        self.data = data
        self.image_size = image_size
        self.index = 0 
        self.on_epoch_end()
        print(len(self.data))

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        self.index = idx % len(self)
        batch_data = self.data.iloc[self.index * self.batch_size : (self.index + 1) * self.batch_size]
        batch_images = []
        batch_labels = []

        for i, row in batch_data.iterrows():
            image_path = os.path.join(self.image_dir, row[INDEX_COLUM_NAME])
            label = row[LABEL_COLUMN_NAME]

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Preprocesing danych
            image = cv2.equalizeHist(image)

            image = np.array(cv2.resize(image, self.image_size))
            image = image.astype('float32') / 255.0  

            batch_images.append(np.expand_dims(image, axis=-1)) 
            batch_labels.append(1 if label == DISEASE else 0)

        return np.array(batch_images), np.array(batch_labels)
    
    def on_epoch_end(self):
        self.data = self.data.sample(frac=1).reset_index(drop=True)

# Przykładowe użycie generatora danych

if __name__ =="__main__":
    df = pd.read_csv('../../sample_labels.csv')
    data = df[df[LABEL_COLUMN_NAME].isin([DISEASE, "No Finding"])][["Image Index", "Finding Labels"]]

    filtered_df = df[df[LABEL_COLUMN_NAME].isin([DISEASE, "No Finding"])][["Image Index", "Finding Labels"]]

    disease_df = filtered_df[filtered_df[LABEL_COLUMN_NAME] == DISEASE]
    no_finding_df = filtered_df[filtered_df[LABEL_COLUMN_NAME] == "No Finding"]
    
    disease_count = len(disease_df)
    
    no_finding_sampled_df = no_finding_df.sample(n=disease_count, random_state=42)
    
    data = pd.concat([disease_df, no_finding_sampled_df])


    image_dir = '../../images'
    batch_size = 32

    data_generator = DataGenerator(data, image_dir, batch_size)
    batch_images, batch_labels = data_generator[8]

    # Wyświetlenie kilku przykładowych obrazów z etykietami
    plt.figure(figsize=(10, 8))
    for i in range(batch_size):
        plt.subplot(8, 4, i + 1)
        plt.imshow(batch_images[i].squeeze(), cmap='gray')
        plt.title(batch_labels[i])
        plt.axis('off')
    plt.show()

