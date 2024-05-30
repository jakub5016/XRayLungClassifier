import pandas as pd
import matplotlib.pyplot as plt

def get_labels():
    df = pd.read_csv("data.csv")
    labels = df["Labels"]

    # Choroby jakie rozpoznajemy
    # Infiltrative - Naciekowe choroby płuc
    # Effusion - Wysięk płucny
    # Atelectasis - Niedodma płuc 
    # Nodule - guzek płuc
    # Lung Consolidation - zagęszczenie tkanki płucnej wskutek nacieku 
    # Pneumothorax  - odma płucna

    labels_using = ["Infiltration", "Effusion", "Atelectasis", "Nodule", "Consolidation", "Pneumothorax"]

    unique_labels = set()
    for label in labels:
        
        multiple_disise = False
        for letter in label:
            if letter == "|":
                multiple_disise = True
                break
        

        if not multiple_disise:
            for using in labels_using:
                if using == label:
                    unique_labels.add(label)
        else:
            disise_to_add = ''
            disise_arr = label.split("|")
            disise_arr.sort()
            for i in disise_arr:
                for j in labels_using:
                    if j == i:
                        disise_to_add += i + "|"

            added = False
            for i in disise_arr:
                if added == False:
                    for j in labels_using:
                        if j == i:
                            unique_labels.add(disise_to_add[:len(disise_to_add)-1])
                            added = True

    unique_labels = list(unique_labels)
    unique_labels.sort()

    return unique_labels


if __name__ =="__main__":
    unique_labels = get_labels()

    [print(i) for i in unique_labels]
