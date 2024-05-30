import pandas as pd
import shutil
import os

LABEL_COLUMN_NAME = "Finding Labels"
INDEX_COLUM_NAME = "Image Index"
DISEASE = "Pneumothorax"

if __name__ =="__main__":
    df = pd.read_csv('../../sample_labels.csv')

    list_of_pictures = df[df[LABEL_COLUMN_NAME].isin([DISEASE, "No Finding"])]["Image Index"].tolist()
    
    os.mkdir("./images")
    for picture_name in list_of_pictures:
        shutil.copy("../../images/" + picture_name, "./images")


    print(df)
