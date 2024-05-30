import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

LABEL_COLUMN_NAME = "Finding Labels"
INDEX_COLUM_NAME = "Image Index"
DISEASE = "Pneumothorax"

if __name__ =="__main__":
    df = pd.read_csv('../../sample_labels.csv')

    df = df[df[LABEL_COLUMN_NAME].str.contains(DISEASE)]

    print(f"Ammout of {DISEASE} records: {df.size}")

    images_array =[cv2.imread("../../images/" + i) for i in df[INDEX_COLUM_NAME].head(4)]

    images_array = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images_array]
    images_array_equilzed = [cv2.equalizeHist(image) for image in images_array]

    num_images = len(images_array)

    # Określenie liczby kolumn i wierszy do wyświetlania obrazów
    cols = 2  # Na przykład 2 kolumny
    rows = math.ceil(num_images / cols)

    fig, axs = plt.subplots(rows, cols * 2, figsize=(15, 5 * rows))
    
    for i, (img, img_equalized) in enumerate(zip(images_array, images_array_equilzed)):
        row = i // cols
        col = i % cols

        axs[row, col * 2].imshow(img, cmap='gray')

        axs[row, col * 2 + 1].imshow(img_equalized, cmap='gray')


        # # Oblicz histogram
        # hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        # axs[row, col * 2 + 1].plot(hist)
        # axs[row, col * 2 + 1].set_xlim([0, 256])
        # axs[row, col * 2 + 1].set_ylim([0, max(hist)])
        # axs[row, col * 2 + 1].set_title("Histogram")

    # Ukrycie niewykorzystanych osi
    for j in range(i + 1, rows * cols):
        row = j // cols
        col = j % cols
        axs[row, col * 2].axis('off')
        axs[row, col * 2 + 1].axis('off')

    plt.tight_layout()
    plt.show()