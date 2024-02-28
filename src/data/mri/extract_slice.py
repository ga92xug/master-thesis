from typing import Dict, Union
import pandas as pd
import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt

PATH = "~/Data/frischs/datasets/adni/"

def extract_slice_from_data(
        data: np.ndarray, 
        slice_num: int, 
        view: str
    ):

    # Extract the specified slice based on the view
    if view == 'axial':
        extracted_slice = data[:, :, slice_num]
    elif view == 'sagittal':
        extracted_slice = data[:, slice_num, :]
    elif view == 'coronal':
        extracted_slice = data[slice_num, :, :]
    else:
        raise ValueError("Invalid view specified")

    return extracted_slice


def extract_slices(
        file_path: str,
        use_seg_mask: bool = True,
        axial_slice_num: int = 52,
        sagittal_slice_num: int = 58,
        coronal_slice_num: int = 92,
    ):
    

    folder_elements = os.listdir(file_path)
    assert len(folder_elements) == 6, f"Folder {file_path} does not contain 5 elements. It contains {folder_elements}"

    # post-processed MRI image
    post_processed_mri_image = [os.path.join(file_path, file) for file in folder_elements if "transformation" not in file and "segm" not in file]
    assert len(post_processed_mri_image) == 1, f"Folder {file_path} does not contain 1 post-processed MRI image. It contains {post_processed_mri_image}"
    post_processed_mri_image = nib.load(post_processed_mri_image[0]).get_fdata()

    # segmentation
    if use_seg_mask:
        segm = [os.path.join(file_path, file) for file in folder_elements if "segm" in file]
        assert len(segm) == 3, f"Folder {file_path} does not contain 3 segmentation. It contains {segm}"
        # get data
        segm = [nib.load(file_path).get_fdata() for file_path in segm]
        # build mask by stacking and summing
        segm_mask = np.stack(segm, axis=0).sum(axis=0)
        # apply mask
        post_processed_mri_image = np.ma.masked_where(segm_mask == 0, post_processed_mri_image)
        post_processed_mri_image = np.ma.filled(post_processed_mri_image, 0)

    # extract slices
    slices = {
        f"axial-{axial_slice_num}": extract_slice_from_data(post_processed_mri_image, axial_slice_num, 'axial'),
        f"sagittal-{sagittal_slice_num}": extract_slice_from_data(post_processed_mri_image, sagittal_slice_num, 'sagittal'),
        f"coronal-{coronal_slice_num}": extract_slice_from_data(post_processed_mri_image, coronal_slice_num, 'coronal'),
    }
    return slices


def plot_slices(slices: Dict[str, np.ndarray]):
    axial_key = [key for key in slices.keys() if "axial" in key][0]
    sagittal_key = [key for key in slices.keys() if "sagittal" in key][0]
    coronal_key = [key for key in slices.keys() if "coronal" in key][0]
    axial = slices[axial_key]
    sagittal = slices[sagittal_key]
    coronal = slices[coronal_key]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    #axes = axes.flatten()
    axes[0].set_title(f'Axial Slice')
    axes[1].set_title(f'Sagittal Slice')
    axes[2].set_title(f'Coronal Slice')

    axes[0].imshow(axial.T, cmap='gray', origin='lower')
    axes[1].imshow(sagittal.T, cmap='gray', origin='lower')
    axes[2].imshow(coronal.T, cmap='gray', origin='lower')

    plt.tight_layout
    plt.show()
    


# Function to load and extract a specific slice
def extract_slice(file_path, slice_num, view):
    # Load the NIfTI file
    img = nib.load(file_path)
    data = img.get_fdata()

    # Extract the specified slice based on the view
    if view == 'axial':
        extracted_slice = data[:, :, slice_num]
    elif view == 'sagittal':
        extracted_slice = data[:, slice_num, :]
    elif view == 'coronal':
        extracted_slice = data[slice_num, :, :]
    else:
        raise ValueError("Invalid view specified")

    # if slice has 3 dimensions, convert to 2D
    if 1 in extracted_slice.shape:
        extracted_slice = np.squeeze(extracted_slice)

    if not (len(extracted_slice.shape) == 2
        or len(extracted_slice.shape) == 3 and extracted_slice.shape[0] in [3, 4]):

        if 3 in extracted_slice.shape:
            # where the last dimension is 3, try to reorder
            extracted_slice = np.swapaxes(extracted_slice, 0, 2)
            #extracted_slice = extracted_slice.transpose(2, 0, 1)
  
    return extracted_slice


def is_t3(session_path: str) -> bool:
    # check for tsv file and read it into pandas
    tsv_file = [file for file in os.listdir(session_path) if file.endswith(".tsv")]
    assert len(tsv_file) == 1, f"Folder {session_path} does not contain 1 tsv file. It contains {tsv_file}"
    tsv_file = os.path.join(session_path, tsv_file[0])
    tsv = pd.read_csv(tsv_file, sep="\t")

    # check that the tsv file contains exactly 1 row
    assert tsv.shape[0] == 1, f"File {tsv_file} does not contain 1 row. It contains {tsv.shape[0]} rows"
    # check that mri scan type is 
    if tsv["mri_field"].values[0] == 1.5:
        return False
    elif tsv["mri_field"].values[0] == 3:
        return True
    else:
        raise ValueError(f"Invalid mri_field value: {tsv['mri_field'].values[0]}")