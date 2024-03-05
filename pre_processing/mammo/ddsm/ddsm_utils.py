import os
import pandas as pd

def data_cleaning(df: pd.DataFrame):
    df = df.rename(columns={
        'image file path':'image_file_path',
        'cropped image file path': 'cropped_image_file_path',
        'mass shape':'mass_shape',
        'left or right breast':'left_or_right_breast',
        'image view':'image_view',
        'abnormality type':'abnormality_type'
    })
    #mass_df['left_or_right_breast'] = mass_df['left_or_right_breast'].astype('category')
    #mass_df['image_view'] = mass_df['image_view'].astype('category')
    #mass_df['mass_margins'] = mass_df['mass_margins'].astype('category')
    #mass_df['mass_shape'] = mass_df['mass_shape'].astype('category')
    #mass_df['abnormality_type'] = mass_df['abnormality_type'].astype('category')
    #mass_df['pathology'] = mass_df['pathology'].astype('category')
    #mass_df_copy.isna().sum()

    #mass_df_copy['mass_shape'].fillna(method = 'bfill', axis = 0, inplace=True) 
    #mass_df_copy['mass_margins'].fillna(method = 'bfill', axis = 0, inplace=True) 
    #mass_df_copy.isna().sum()
    return df


def dicom_data_cleaning(dicom_df: pd.DataFrame):
    #dicom_df_copy = dicom_df.copy()

    dicom_df.drop(['PatientBirthDate','AccessionNumber','Columns',
        'ContentDate','ContentTime','PatientSex','PatientBirthDate',
        'ReferringPhysicianName','Rows','SOPClassUID','SOPInstanceUID',
        'StudyDate','StudyID','StudyInstanceUID','StudyTime','InstanceNumber',
        'SeriesInstanceUID','SeriesNumber'],axis =1, inplace=True) 

    #dicom_df_copy['SeriesDescription'].fillna(method = 'bfill', axis = 0, inplace=True)
    #dicom_df_copy['Laterality'].fillna(method = 'bfill', axis = 0, inplace=True)
    return dicom_df

def mass_data_cleaning(mass_df: pd.DataFrame):
    mass_df = mass_df.rename(columns={
        'image file path':'image_file_path',
        'cropped image file path': 'cropped_image_file_path',
        'mass shape':'mass_shape',
        'left or right breast':'left_or_right_breast',
        'mass margins':'mass_margins',
        'image view':'image_view',
        'abnormality type':'abnormality_type'
    })
    mass_df['left_or_right_breast'] = mass_df['left_or_right_breast'].astype('category')
    mass_df['image_view'] = mass_df['image_view'].astype('category')
    mass_df['mass_margins'] = mass_df['mass_margins'].astype('category')
    mass_df['mass_shape'] = mass_df['mass_shape'].astype('category')
    mass_df['abnormality_type'] = mass_df['abnormality_type'].astype('category')
    mass_df['pathology'] = mass_df['pathology'].astype('category')
    #mass_df_copy.isna().sum()

    #mass_df_copy['mass_shape'].fillna(method = 'bfill', axis = 0, inplace=True) 
    #mass_df_copy['mass_margins'].fillna(method = 'bfill', axis = 0, inplace=True) 
    #mass_df_copy.isna().sum()
    return mass_df



def calc_data_cleaning(calc_df: pd.DataFrame):
    #calc_df_copy = calc_df.copy()
    calc_df = calc_df.rename(columns={'calc type':'calc_type'})
    calc_df = calc_df.rename(columns={'calc distribution':'calc_distribution'})
    calc_df = calc_df.rename(columns={'image view':'image_view'})
    calc_df = calc_df.rename(columns={'left or right breast':'left_or_right_breast'})
    calc_df = calc_df.rename(columns={'breast density':'breast_density'})
    calc_df = calc_df.rename(columns={'abnormality type':'abnormality_type'})
    calc_df['pathology'] = calc_df['pathology'].astype('category')
    calc_df['calc_type'] = calc_df['calc_type'].astype('category')
    calc_df['calc_distribution'] = calc_df['calc_distribution'].astype('category')
    calc_df['abnormality_type'] = calc_df['abnormality_type'].astype('category')
    calc_df['image_view'] = calc_df['image_view'].astype('category')
    calc_df['left_or_right_breast'] = calc_df['left_or_right_breast'].astype('category')
    #calc_df.isna().sum()
    #calc_df['calc_type'] = calc_df['calc_type'].fillna(method = 'bfill', axis = 0,) 
    #calc_df['calc_distribution'] = calc_df['calc_distribution'].fillna(method = 'bfill', axis = 0)
    #calc_df.isna().sum()
    return calc_df