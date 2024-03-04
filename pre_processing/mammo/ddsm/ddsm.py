import os
from typing import Dict
import pandas as pd
import os
import sys

from sklearn.model_selection import train_test_split
sys.path.append(os.getcwd())

from pre_processing.ddsm_utils import mass_data_cleaning

def build_dicts(full_mammogram_images, cropped_images, ROI_mask_images):
    # organize image paths
    full_mammo_dict = dict()
    cropped_images_dict = dict()
    roi_img_dict = dict()

    for dicom in full_mammogram_images:
        key = dicom.split("/")[-2]
        full_mammo_dict[key] = dicom
    for dicom in cropped_images:
        key = dicom.split("/")[-2]
        cropped_images_dict[key] = dicom
    for dicom in ROI_mask_images:
        key = dicom.split("/")[-2]
        roi_img_dict[key] = dicom

    return full_mammo_dict, cropped_images_dict, roi_img_dict

def get_calc_data(base_path: str, mode: str):
    assert mode in ['train', 'test']
    path = os.path.join(base_path, f'csv/calc_case_description_{mode}_set.csv')
    calc_df = pd.read_csv(path)
    return calc_df

def get_mass_data(
        base_path: str, 
        mode: str, 
        full_mammo_dict: Dict[str, str], 
        cropped_images_dict: Dict[str, str], 
        roi_img_dict: Dict[str, str]
    ):
    assert mode in ['train', 'test']
    path = os.path.join(base_path, f'csv/mass_case_description_{mode}_set.csv')
    mass_df = pd.read_csv(path)
    fix_image_path(mass_df, full_mammo_dict, cropped_images_dict, roi_img_dict)
    mass_df = mass_data_cleaning(mass_df)

    class_mapper = {'MALIGNANT': "malignant", 'BENIGN': 'begnign', 'BENIGN_WITHOUT_CALLBACK': 'begnign'} 
    mass_df['label'] = mass_df['pathology'].map(class_mapper)

    return mass_df[["patient_id", "image_file_path", "label"]]

# fix image paths
def fix_image_path(df, full_mammo_dict, cropped_images_dict, roi_img_dict):
    """correct dicom paths to correct image paths"""
    for index, img in enumerate(df.values):
        img_name = img[11].split("/")[2]
        df.iloc[index,11] = full_mammo_dict[img_name]
        img_name = img[12].split("/")[2]
        df.iloc[index,12] = cropped_images_dict[img_name]
        img_name = img[13].split("/")[2]
        df.iloc[index,13] = roi_img_dict[img_name]

def split_stratified_with_person_id(df: pd.DataFrame, split: float = 0.75):
    """split train data into train and test data stratified by label but considering person_id"""
    # Get unique patient IDs
    unique_patient_ids = df[['patient_id', "label"]].drop_duplicates(subset=['patient_id'], keep='first')
    # 13 women have a cancer in only one breast and a benign in the other
    assert unique_patient_ids.shape[0] + 13 == df[['patient_id', "label"]].drop_duplicates(subset=['patient_id', "label"], keep='first').shape[0]

    # Split patient IDs into train and validation sets
    train_patient_ids, valid_patient_ids = train_test_split(unique_patient_ids, train_size=split, stratify=unique_patient_ids['label'], random_state=42)
    train_patient_ids = train_patient_ids["patient_id"].to_list()
    valid_patient_ids = valid_patient_ids["patient_id"].to_list()

    # check that no patient is in both sets
    assert len(set(train_patient_ids).intersection(set(valid_patient_ids))) == 0

    # Step 3: Filter DataFrame based on train and validation patient IDs
    train_df = df[df['patient_id'].isin(train_patient_ids)]
    valid_df = df[df['patient_id'].isin(valid_patient_ids)]

    # reset index for the new DataFrames
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)

    return train_df, valid_df


def get_ddsm_df(ddsm_path: str):
    image_dir = os.path.join(ddsm_path, 'jpeg')

    # dicom data
    dicom_data = pd.read_csv(os.path.join(ddsm_path, 'csv/dicom_info.csv'))
    dicom_data['image_path'] = dicom_data.image_path.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
    cropped_images = dicom_data[dicom_data.SeriesDescription == 'cropped images'].image_path
    full_mammogram_images = dicom_data[dicom_data.SeriesDescription == 'full mammogram images'].image_path
    ROI_mask_images = dicom_data[dicom_data.SeriesDescription == 'ROI mask images'].image_path
    full_mammo_dict, cropped_images_dict, roi_img_dict = build_dicts(full_mammogram_images, cropped_images, ROI_mask_images)

    #calc_train, calc_test = get_calc_data(ddsm_path, 'train'), get_calc_data(ddsm_path, 'test')
    mass_train = get_mass_data(ddsm_path, 'train', full_mammo_dict, cropped_images_dict, roi_img_dict)
    mass_test = get_mass_data(ddsm_path, 'test', full_mammo_dict, cropped_images_dict, roi_img_dict)

    mass_train, mass_valid = split_stratified_with_person_id(mass_train, split=0.8)
    
    dataset_dict = {
        "train": mass_train,
        "valid": mass_valid,
        "test": mass_test
    }
    return dataset_dict


def main():
    path = os.path.expanduser("~/Data/frischs/datasets/mammography/")
    mass_train, mass_valid, mass_test = get_ddsm_df(path)
    return mass_train, mass_valid, mass_test


if __name__ == '__main__':
    main()