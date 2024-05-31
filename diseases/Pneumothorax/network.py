import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import pandas as pd
import numpy as np
import os
from generator import DataGenerator

LABEL_COLUMN_NAME = "Finding Labels"
INDEX_COLUM_NAME = "Image Index"
DISEASE = "Pneumothorax"
SIZE = 256

if __name__ =="__main__":

    df = pd.read_csv('../../sample_labels.csv')
    filtered_df = df[df[LABEL_COLUMN_NAME].isin([DISEASE, "No Finding"])][["Image Index", "Finding Labels"]]

    disease_df = filtered_df[filtered_df[LABEL_COLUMN_NAME] == DISEASE]
    no_finding_df = filtered_df[filtered_df[LABEL_COLUMN_NAME] == "No Finding"]
    
    disease_count = len(disease_df)
    
    no_finding_sampled_df = no_finding_df.sample(n=disease_count, random_state=42)
    
    data = pd.concat([disease_df, no_finding_sampled_df])

    image_dir = '../../images'
    batch_size = 8

    def generator_func(data, image_dir, batch_size):
        gen = DataGenerator(data, image_dir, batch_size)
        while True:
            for i in range(len(gen)):
                yield gen[i]

    train_dataset = tf.data.Dataset.from_generator(
        lambda: generator_func(data, image_dir, batch_size),
        output_signature=(
            tf.TensorSpec(shape=(32, SIZE, SIZE, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(32,), dtype=tf.int32)
        )
    ).repeat()  # Repeat indefinitely
    
    
    model = Sequential([
        Input(shape=(256, 256, 1)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(
        train_dataset,
        steps_per_epoch=5,
        epochs=50
    )

    # Zapisywanie wytrenowanego modelu
    model.save('my_model.h5')