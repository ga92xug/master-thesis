"""
Space on the hard drive is limited so we need to delete the post-processed MRI image after we have extracted the slices.
conda activate clinicaEnv
"""
import argparse
import os
from typing import List
import pandas as pd
import subprocess

from extract_slice import *

# PATH = os.path.expanduser("~/Data/frischs/datasets/adni/adni1")
# BIDS_PATH = os.path.join(PATH, "bids")
# TSV_PATH = os.path.join(PATH, "temp.tsv")
# CAPS_PATH = os.path.join(PATH, "caps")
# SLICE2DDATA = os.path.join(PATH, "slice2Ddata")


class Clinica_Connector:
    def __init__(self, run_pipeline: bool, pipeline: str = "t1-volume-tissue-segmentation", n_threads: int = 8):
        self.run_pipeline = run_pipeline
        self.pipeline = pipeline
        self.clinica_command = [
            "clinica", "run", pipeline, 
            "-np", str(n_threads), "-tsv", TSV_PATH, 
            BIDS_PATH, CAPS_PATH
        ]
        self.converted_path = os.path.join(PATH, "converted_subjects.tsv")
        self.skipped_path = os.path.join(PATH, "skipped_sessions.tsv")
        if not os.path.exists(self.converted_path):
            self.create_tsv_file(self.converted_path)
        if not os.path.exists(self.skipped_path):
            self.create_tsv_file(self.skipped_path, optional="\treason")

        # create a folder to store the slices
        os.makedirs(SLICE2DDATA, exist_ok=True)
        self.label_df = pd.read_csv(os.path.join(PATH, "bids/participants.tsv"), sep="\t")

    def create_tsv_file(self, path, optional: str = ""):
        # create tsv file
        with open(path, "w") as f:
            f.write(f"participant_id\tsession_id{optional}\n")

    def append_to_tsv_file(self, path, subject: str, session: str, optional: str = ""):
        with open(path, "a") as f:
            f.write(f"{subject}\t{session}{optional}\n")

    def read_tsv_file(self, path):
        df = pd.read_csv(path, sep="\t")
        return df   

    def check_if_already_converted(self, subject: str, session: str):
        # check if the session is already converted
        converted_df = self.read_tsv_file(self.converted_path)
        skipped_df = self.read_tsv_file(self.skipped_path)

        # check if there is a row with the same subject and session
        if len(converted_df[(converted_df["participant_id"] == subject) & (converted_df["session_id"] == session)]) > 0:
            return True
        if len(skipped_df[(skipped_df["participant_id"] == subject) & (skipped_df["session_id"] == session)]) > 0:
            return True
        return False

    def add_to_pipeline_queue(self, subject: str, session: str):
        # check if the session is T3
        session_path = os.path.join(BIDS_PATH, subject, session)
        if is_t3(session_path):
            self.append_to_tsv_file(self.skipped_path, subject, session, optional="\tT3")
            print(f"Skipped so far {len(self.read_tsv_file(self.skipped_path))} sessions")
            return

        # append to tsv file
        self.append_to_tsv_file(TSV_PATH, subject, session)


    def run(self):
        # check that the tsv file contains at least 1 row
        current_df = self.read_tsv_file(TSV_PATH)
        if len(current_df) < 1:
            return

        if not self.run_pipeline:
            print("\tPipeline would run:", current_df.iloc[0]["participant_id"], current_df.iloc[0]["session_id"])
        else:
            subprocess.run(self.clinica_command)
            # check if the pipeline ran successfully
            try: 
                # extract 2D images from the caps folder
                self.extract_2D_images_from_caps()  
                # add session to converted sessions file
                self.append_to_tsv_file(self.converted_path, current_df.iloc[0]["participant_id"], current_df.iloc[0]["session_id"])
                subprocess.run(["rm", "-r", CAPS_PATH])
            except:
                # add session to skipped sessions file
                self.append_to_tsv_file(self.skipped_path, current_df.iloc[0]["participant_id"], current_df.iloc[0]["session_id"], optional="\tpipeline_failed")

        # delete the tsv file and the CAPS folder
        os.remove(TSV_PATH)

    def finish(self):
        # print the number of skipped sessions
        if os.path.exists(self.skipped_path):
            skipped_df = self.read_tsv_file(self.skipped_path)
            print(f"Skipped {len(skipped_df)} sessions")

        # print the number of converted subjects
        if self.run_pipeline:
            converted_df = self.read_tsv_file(self.converted_path)
            print(f"Converted {len(converted_df)} MRI scans")

    def extract_2D_images_from_caps(
        self,
        use_seg_mask: bool = True,
        axial_slice_num: int = 52,
        sagittal_slice_num: int = 58,
        coronal_slice_num: int = 92,
    ):
        # loop through all subjects in the CAPS folder
        subjects_folder = os.path.join(CAPS_PATH, "subjects")
        for subject in os.listdir(subjects_folder):
            if "sub-ADNI" not in subject:
                continue
            subject_path = os.path.join(subjects_folder, subject)
            # loop through all sessions
            for session in os.listdir(subject_path):
                if "ses-M" not in session:
                    continue

                slices = extract_slices(
                    file_path=os.path.join(
                        subject_path, session, "t1", "spm", "segmentation", "normalized_space"
                    ),
                    use_seg_mask=use_seg_mask,
                    axial_slice_num=axial_slice_num,
                    sagittal_slice_num=sagittal_slice_num,
                    coronal_slice_num=coronal_slice_num,
                )

                self.save_slices_to_disk(slices, subject, session)

    def save_slices_to_disk(self, slices: Dict[str, np.ndarray], subject: str, session: str):
        # label based on the participants.tsv file with the diagnosis column
        label = self.label_df[self.label_df["participant_id"] == subject]["diagnosis_sc"].values[0]
        path = os.path.join(SLICE2DDATA, label)
        os.makedirs(path, exist_ok=True)

        # save slices to disk as numpy arrays
        for slice_name, slice_data in slices.items():
            slice_path = os.path.join(path, f"{slice_name}_{subject}_{session}.npy")
            with open(slice_path, "wb") as f:
                np.save(f, slice_data)    


def append_to_file(file_path: str, data: str):
    with open(file_path, "a") as f:
        f.write(data + "\n")


def main(run_pipeline: bool, path: str, filter_type: str):
    global PATH, BIDS_PATH, TSV_PATH, CAPS_PATH, SLICE2DDATA
    PATH = os.path.expanduser(path)
    BIDS_PATH = os.path.join(PATH, "bids")
    TSV_PATH = os.path.join(PATH, "temp.tsv")
    CAPS_PATH = os.path.join(PATH, "caps")
    SLICE2DDATA = os.path.join(PATH, "slice2Ddata")

    # create clinica connector
    clinica_connector = Clinica_Connector(run_pipeline=run_pipeline)

    # loop through all subjects
    i = 0
    for subject in os.listdir(BIDS_PATH):
        if "sub-ADNI" not in subject:
            continue
        print(f"Subject {i}: {subject}")
        
        # loop through all sessions
        j = 0
        for session in os.listdir(os.path.join(BIDS_PATH, subject)):
            if "tsv" in session:
                continue
            print(f"\tSession {j}: {session}")
            already_converted = clinica_connector.check_if_already_converted(subject, session)
            if already_converted:
                print(f"\t Already converted")
                j += 1
                continue
            
            clinica_connector.create_tsv_file(TSV_PATH)
            # append to tsv file
            clinica_connector.add_to_pipeline_queue(subject, session)
            
            #if j >= 1:
            #    break
            j += 1
        
            # run the pipeline
            clinica_connector.run()

        #if i > 1:
        #    break
        i += 1

    # finish the pipeline
    clinica_connector.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run postprocessing pipeline for MRI images.")
    parser.add_argument("--run_pipeline", type=bool, default=True, help="Whether to run the pipeline or not.")
    parser.add_argument("--path", type=str, required=True, help="Base path for the data directories.")
    parser.add_argument("--filter_type", type=str, choices=['T3', 'T1.5'], required=True, help="Filter for T3 or T1.5 MRI types.")

    args = parser.parse_args()

    main(run_pipeline=args.run_pipeline, path=args.path, filter_type=args.filter_type)