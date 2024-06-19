"""
File: 4.1.3_get_seg_hpc.py
Description: This loads the already created segmentation files from the HPC.
Author: pvondrlik
Version: 1.0
Last Updated: 2024-03-08
"""

import pandas as pd
import subprocess
import subprocess

folder_path = "/data/"  # path to the data folder
eval_path_results = "/eval_frames_label.csv"  # path to the evaluation file
data = pd.read_csv(eval_path_results)

for i in range(0, len(data)):
    x = data["x"].iloc[i]
    y = data["y"].iloc[i]
    session = data["session"].iloc[i]
    response = data["response"].iloc[i]
    frame_nr = data["frame_nr"].iloc[i]
    str_frame_nr = f"{frame_nr:05d}"

    # get image path
    img_path = (
        folder_path + session + "/video_frames_img/frame_" + str_frame_nr + ".jpg"
    )

    command = [
        "scp",
        "-r",
        "name@cluster.de:/data/evalset/segs/"
        + session
        + "/Segmasks/frame_"
        + str_frame_nr
        + ".pt",
        "/data/evalset/segs/" + session + "_" + str_frame_nr + ".pt",
    ]

    # Call the command in the terminal
    subprocess.run(command)
