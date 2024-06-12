#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: 4.1.2_get_frames_from_remote.py
Description: This script retrieves video frames from a remote server using SCP (Secure Copy Protocol).
Call: python3 4.1.1_get_frames_from_remote.py
Author: pvondrlik
Version: 1.0
Last Updated: 2024-03-08
"""

import pandas as pd
import subprocess


def get_frames_from_remote():
    """
    This function retrieves video frames from a remote server using SCP (Secure Copy Protocol).
    It reads a CSV file containing session and frame number information, and for each row in the file,
    it constructs an SCP command to copy the corresponding video frame from the remote server to a local directory.

    Returns:
    None
    """
    # load the csv file
    file_path = "data/cyprus_eval_frames.csv"  # which frames to load
    file = pd.read_csv(file_path)

    for i, row in file.iterrows():
        session = row["session"]
        frame_nr = row["frame_nr"]
        str_frame_nr = f"{frame_nr:05d}"

        fix_df = pd.read_csv(
            "/data/" + session + "/fixation_and_labels_extended.csv"
        )  # file containing the fixation data

        # load all frames
        fix_id = fix_df[fix_df["frame_nr"] == frame_nr].iloc[0]["fixation_id"]
        # nr_frames = fix_df[fix_df["fixation_id"] == fix_id ]["frame_nr"].nunique()
        df_groups = fix_df[fix_df["fixation_id"] == fix_id].groupby("frame_nr")

        for frame_nr, group in df_groups:
            str_frame_nr = f"{frame_nr:05d}"

            command2 = [
                "scp",
                "-r",
                "name@computer.de:/data/"
                + session
                + "/video_frames_img/frame_"
                + str_frame_nr
                + ".jpg",
                "/data/" + session + "/video_frames_img/",
            ]
            result = subprocess.run(command2)


if __name__ == "__main__":
    get_frames_from_remote()
