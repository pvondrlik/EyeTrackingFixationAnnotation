import argparse
import time
import torch
import os
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

# import from my files
import sys

folder_path = ""

sys.path.append(folder_path + "/src")
from features.gaze import get_frame_per_gaze
from features.label import segment_prediction


def main(session_name, start_frame, end_frame):
    print(torch.cuda.is_available())
    print(torch.__version__)
    print(torch.version.cuda)
    torch.cuda.empty_cache()

    fix_path = folder_path + session_name + "/fixations.csv"
    gaze_path = folder_path + session_name + "/gaze.csv"
    world_path = folder_path + session_name + "/world_timestamps.csv"
    input_img_path = folder_path + session_name + "/video_frames_img/"

    result_folder = folder_path
    result_img_path = result_folder + session_name + "/Segimgs/"
    result_seg_path = result_folder + session_name + "/Segmasks/"
    result_label_path = result_folder + session_name + "/Labels/"

    # create folders if not existent
    os.makedirs(result_img_path, exist_ok=True)
    os.makedirs(result_label_path, exist_ok=True)
    os.makedirs(result_seg_path, exist_ok=True)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    # segformer
    path_seg_model = folder_path + "/models/segformer-b3-finetuned-cityscapes-1024-1024"
    processor = SegformerImageProcessor.from_pretrained(
        path_seg_model
    )  # yields different results in feature extractor
    seg_model = SegformerForSemanticSegmentation.from_pretrained(p)

    # get the merged frame
    merged_gaze_df = get_frame_per_gaze(gaze_path, world_path)

    fields = ["session", "frame_nr", "x", "y", "label", "overlap_lbl_mask"]

    # get images
    img_paths = [
        os.path.join(result_img_path, file)
        for file in os.listdir(result_img_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp"))
    ]

    # predict for all images and store the data in csv
    STARTFRAME = int(start_frame)
    ENDFRAME = int(end_frame)

    for frame_nr in range(STARTFRAME, ENDFRAME):
        frame_nr_str = f"{frame_nr:05d}"
        img_path = os.path.join(input_img_path, "frame_" + frame_nr_str + ".jpg")

        if os.path.exists(img_path):
            try:
                logits = segment_prediction(img_path, frame_nr, seg_model, processor)

                mask_file_name = result_seg_path + "frame_" + frame_nr_str + ".pt"
                torch.save(logits, mask_file_name)
                print(
                    f"Saved segmentaion for {frame_nr_str} in {session_name} in {mask_file_name}"
                )

            except ValueError as e:
                print(f"A ValueError occurred: {e}")
                print("Error with: ", session_name, frame_nr)

        else:
            print(f"Path does not exist: {img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process session name.")
    parser.add_argument(
        "--session", type=str, required=True, help="Session name to process."
    )
    parser.add_argument(
        "--start_frame", type=str, required=True, help="Session name to process."
    )
    parser.add_argument(
        "--end_frame", type=str, required=True, help="Session name to process."
    )

    args = parser.parse_args()

    main(args.session, args.start_frame, args.end_frame)
