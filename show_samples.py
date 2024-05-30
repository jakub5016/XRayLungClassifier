import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

LABEL_COLUMN_NAME = "Finding Labels"
INDEX_COLUM_NAME = "Image Index"


if __name__ =="__main__":
    df = pd.read_csv('sample_labels.csv')

    labels = ["Infiltration", "Effusion", "Atelectasis", "Nodule", "Consolidation", "Pneumothorax"] # Infiltrative / Infiltration

    print(f"Ammout of elements before filter: {df.size}")

    df = df[df[LABEL_COLUMN_NAME].isin(labels)]

    print(f"Ammout of elements before after: {df.size}")

    examples = df.groupby(LABEL_COLUMN_NAME).first().reset_index()

    print(examples) # Pierwsze przypadki w bazie danych, dla każdej choroby

    images_array =[cv2.imread("images/" + i) for i in examples[INDEX_COLUM_NAME]]

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
        axs[row, col].set_title(examples[LABEL_COLUMN_NAME][i])

    # Ukrycie niewykorzystanych osi
    for j in range(i + 1, rows * cols):
        row = j // cols
        col = j % cols
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.show()

