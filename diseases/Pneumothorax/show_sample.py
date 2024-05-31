import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os

LABEL_COLUMN_NAME = "Finding Labels"
INDEX_COLUM_NAME = "Image Index"
DISEASE = "Pneumothorax"


if __name__ =="__main__":
    df = pd.read_csv('../../sample_labels.csv')

    df = df[df[LABEL_COLUMN_NAME].isin([DISEASE])]
    

    print(f"Ammout of {DISEASE} records: {df.size}")

    images_array =[cv2.imread("../../images/" + i) for i in df[INDEX_COLUM_NAME].head(9)]

    num_images = len(images_array)

    # Określenie liczby kolumn i wierszy do wyświetlania obrazów
    cols = 3  # Na przykład 3 kolumny
    rows = math.ceil(num_images / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(15, 5))
    for i, img in enumerate(images_array):
        row = i // cols
        col = i % cols
        axs[row, col].imshow(img, cmap='gray')
        axs[row, col].axis('off')

    # Ukrycie niewykorzystanych osi
    for j in range(i + 1, rows * cols):
        row = j // cols
        col = j % cols
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.show()
