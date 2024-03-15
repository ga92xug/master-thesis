import os
from glob import glob
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split


def get_derm7pt_df(base_path: str) -> pd.DataFrame:
    """
    Adpated from :
    - https://arxiv.org/abs/2101.03814 -> https://github.com/j05t/lesion-analysis/blob/master/isic_classifier_nasnetalarge.ipynb
    """
    derm7pt_path = os.path.join(base_path, "derm7pt", 'release_v0/')

    df = pd.read_csv(derm7pt_path + 'meta/meta.csv', usecols=['derm', 'diagnosis'])
    df.rename(columns={'derm': 'name', 'diagnosis': 'label'}, inplace=True)
    
    df.loc[df['label'] == 'basal cell carcinoma', 'label'] = 'BCC'
    df.loc[df['label'] == 'lentigo', 'label'] = 'BKL'
    df.loc[df['label'] == 'seborrheic keratosis', 'label'] = 'BKL'
    df.loc[df['label'] == 'dermatofibroma', 'label'] = 'DF'
    df.loc[df['label'] == 'vascular lesion', 'label'] = 'VASC'
    # https://meshb.nlm.nih.gov/record/ui?ui=D008548
    df.loc[df['label'] == 'melanosis', 'label'] = 'BKL'
    # classified as benign melanocytic lesion
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4866625/
    df.loc[df['label'] == 'reed or spitz nevus', 'label'] = 'NV'
    # Melanocytic nevus
    df.loc[df['label'] == 'blue nevus', 'label'] = 'NV'
    df.loc[df['label'] == 'clark nevus', 'label'] = 'NV'
    df.loc[df['label'] == 'combined nevus', 'label'] = 'NV'
    df.loc[df['label'] == 'congenital nevus', 'label'] = 'NV'
    df.loc[df['label'] == 'dermal nevus', 'label'] = 'NV'
    df.loc[df['label'] == 'melanoma', 'label'] = 'MEL'
    df.loc[df['label'] == 'melanoma metastasis', 'label'] = 'MEL'
    df.loc[df['label'] == 'melanoma (in situ)', 'label'] = 'MEL'
    df.loc[df['label'] == 'melanoma (less than 0.76 mm)', 'label'] = 'MEL'
    df.loc[df['label'] == 'melanoma (0.76 to 1.5 mm)', 'label'] = 'MEL'
    df.loc[df['label'] == 'melanoma (more than 1.5 mm)', 'label'] = 'MEL'
    # none of the others
    df.loc[df['label'] == 'miscellaneous', 'label'] = 'UNK'
    df.loc[df['label'] == 'recurrent nevus', 'label'] = 'UNK'
    
    df = df[['name', 'label']]
    df['name'] = df['name'].apply(lambda x: "{}images/{}".format(derm7pt_path,x))

    # drop rows with UNK label
    df = df[df['label'] != 'UNK']
    return df


def get_ham10000_df(base_path: str) -> pd.DataFrame:
    """
    Adapted from https://www.kaggle.com/code/sid321axn/step-wise-approach-cnn-model-77-0344-accuracy
    """
    ham_dir = os.path.join(base_path, "ham10000")

    # Merging images from both folders HAM10000_images_part1.zip and HAM10000_images_part2.zip into one dictionary
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                        for x in glob(os.path.join(ham_dir, '*', '*.jpg'))}

    # This dictionary is useful for displaying more human-friendly labels later on
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    df = pd.read_csv(os.path.join(ham_dir, 'HAM10000_metadata.csv'))

    # Creating New Columns for better readability
    df['name'] = df['image_id'].map(imageid_path_dict.get)
    # upper case to match the other datasets
    df['label'] = df['dx'].apply(lambda x: x.upper()) 
    # .map(lesion_type_dict.get) 

    df = df[['name', 'label']]
    return df
   

def get_derma_images_labels(base_path: str, val_on: str, test_on: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Train images are always just the ham10000 dataset.
    Val and test can be either ham10000 or derm7pt.

    We always split the datasets the same even if some of the data is not used this way to allow for a fair test comparison.
    """
    assert val_on in ["ham10000", "derm7pt"], "val_on has to be either 'ham10000' or 'derm7pt'"
    assert test_on in ["ham10000", "derm7pt"], "test_on has to be either 'ham10000' or 'derm7pt'"

    # ham10000
    df_ham = get_ham10000_df(base_path)
    train_ham, test_ham = train_test_split(df_ham, test_size=0.3, random_state=42, stratify=df_ham["label"])
    val_ham, test_ham = train_test_split(test_ham, test_size=0.5, random_state=42, stratify=test_ham["label"])

    if val_on == "derm7pt" or test_on == "derm7pt":
        df_derm7pt = get_derm7pt_df(base_path)
        val_derm7pt, test_derm7pt = train_test_split(df_derm7pt, test_size=0.5, random_state=42, stratify=df_derm7pt["label"])

    if val_on == "ham10000":
        val = val_ham
    else:
        val = val_derm7pt

    if test_on == "ham10000":
        test = test_ham
    else:
        test = test_derm7pt

    images = {
        "train": train_ham["name"].tolist(),
        "val": val["name"].tolist(),
        "test": test["name"].tolist(),
    }

    # encode the labels
    label_encoder = LabelEncoder()
    # fit on the train set as HAM10000 has more labels
    train_ham["label"] = label_encoder.fit_transform(train_ham["label"])
    val["label"] = label_encoder.transform(val["label"])
    test["label"] = label_encoder.transform(test["label"])

    labels = {
        "train": train_ham["label"].tolist(),
        "val": val["label"].tolist(),
        "test": test["label"].tolist(),
    }

    return images, labels


if __name__ == "__main__":
    base_path = os.path.expanduser("~/Data/frischs/datasets/")
    df = get_derm7pt_df(base_path)
    print(df["label"].value_counts())
    df = get_ham10000_df(base_path)
    print(df["label"].value_counts())

    images, labels = get_derma_images_labels(base_path, "ham10000", "derm7pt")

    print("images: ", images["train"][0])
    print("labels: ", labels["train"][0])
    print("images: ", images["val"][0])
    print("labels: ", labels["val"][0])
    print("images: ", images["test"][0])
    print("labels: ", labels["test"][0])
