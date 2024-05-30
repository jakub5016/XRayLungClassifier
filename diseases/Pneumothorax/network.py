import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os

LABEL_COLUMN_NAME = "Finding Labels"
INDEX_COLUM_NAME = "Image Index"
DISEASE = "Pneumothorax"

if __name__ =="__main__":
    df = pd.read_csv('../../sample_labels.csv')

    df = df[df[LABEL_COLUMN_NAME].isin(DISEASE)]