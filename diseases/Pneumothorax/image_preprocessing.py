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

    df = df[df[LABEL_COLUMN_NAME].isin([DISEASE])]

    print(df.head(6))
    
    print(f"Ammout of {DISEASE} records: {df.size}")

    images_array =[cv2.imread("../../images/" + i) for i in df[INDEX_COLUM_NAME].head(6)]
    images_array = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images_array]
    
    # Preprocesing
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    images_array_equilzed = [cv2.equalizeHist(image) for image in images_array]
    # images_array_equilzed = [clahe.apply(image) for image in images_array]
    images_array_denoised = [cv2.GaussianBlur(image, (9,9), 0) for image in images_array_equilzed]
    num_images = len(images_array)

    # Określenie liczby kolumn i wierszy do wyświetlania obrazów
    cols = 2  # Na przykład 2 kolumny
    rows = math.ceil(num_images / cols)

    fig, axs = plt.subplots(rows, cols * 2, figsize=(10, 8))
    
    for i, (img, img_equalized) in enumerate(zip(images_array, images_array_equilzed)):
        row = i // cols
        col = i % cols

        axs[row, col * 2].imshow(img, cmap='gray')

        axs[row, col * 2 + 1].imshow(img_equalized, cmap='gray')


    plt.tight_layout()

    fig2, axs2 = plt.subplots(rows, cols * 2, figsize=(10, 8))
    
    for i, (img_equalized, img_denoised) in enumerate(zip(images_array_equilzed, images_array_denoised)):
        row = i // cols
        col = i % cols

        axs2[row, col * 2].imshow(img_equalized, cmap='gray')

        axs2[row, col * 2 + 1].imshow(img_denoised, cmap='gray')

    plt.show()