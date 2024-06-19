import time  
import datetime
import pandas as pd
import os
import argparse

from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)
from ultralytics import SAM

# Change this
folder_path = ""
SAVE_FQ = 100

# import from my files
import sys

sys.path.append(folder_path + "/src")

from features.gaze import get_frame_per_gaze
from features.label import full_prediction



def main(session, start_frame, end_frame, by_frame_file=False, frame_file_path=None ):
    """
    main function to predict the labels for the given session
    
    Args:
    session: str
        session name
    start_frame: int
        start frame number
    end_frame: int
        end frame number
    by_frame_file: bool
        if the frame numbers are given by a file
    frame_file_path: str
        path to the file with the frame numbers
        
    return:
    None
    
    Outputs:
    csv files with the labels for each frame
    one file per a number of frames defined by SAVE_FQ            
    """

    seg_model_path = folder_path + "/models/segformer-b3-finetuned-cityscapes-1024-1024"
    sam_path = folder_path +"/models/mobile_sam.pt"
    
    # by frame file
    if by_frame_file:
        df = pd.read_csv(frame_file_path)
        idx = int(session)
        session_name = df["session"][idx]
        STARTFRAME = df["start"][idx]
        ENDFRAME = df["end"][idx]
        
    else: 
        STARTFRAME = int(start_frame)
        ENDFRAME = int(end_frame)
        session_name = session

    # define files
    fix_path = folder_path + session_name + "/fixations.csv"
    gaze_path = folder_path + session_name + "/gaze.csv"
    world_path = folder_path + session_name + "/world_timestamps.csv"
    input_img_path = folder_path + session_name + "/video_frames_img/"

    result_folder = folder_path
    result_label_path = result_folder + session_name + "/Labels/"
    result_seg_path = result_folder + session_name + "/Segmasks/"
    track_file_path = result_folder + session_name + "/track_file.txt"

    # create folders if  not existen
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    if not os.path.exists(result_label_path):
        os.makedirs(result_label_path)

    # Segformer model
    processor = SegformerImageProcessor.from_pretrained(
        seg_model_path
    )  
    seg_model = SegformerForSemanticSegmentation.from_pretrained(seg_model_path)

    # sam model
    sam = SAM(sam_path)
  
    # get the merged frame
    merged_gaze_df = get_frame_per_gaze(gaze_path, world_path)

    fields = ["session", "frame_nr", "x", "y", "label", "overlap_lbl_mask", "certainty"]

    # # get images
    # img_paths = [os.path.join(result_img_path , file) for file in os.listdir(result_img_path) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

    for n in range(STARTFRAME, ENDFRAME, SAVE_FQ):
        track_file = open(track_file_path, "a")
        n_str = f"{n :05d}"
        data_list = []
        start_time = time.time()
        for frame_nr in range(n, n + SAVE_FQ):
            if frame_nr >= ENDFRAME:
                break

            start_time_frame = time.time()
            frame_nr_str = f"{frame_nr :05d}"
            
            # Extract frame numbers from each string
            img_path = os.path.join(input_img_path, "frame_" + frame_nr_str + ".jpg")
            seg_path = os.path.join(result_seg_path, "frame_" + frame_nr_str + ".pt")

            if os.path.exists(img_path) and os.path.exists(seg_path):
                # get data
                try:
                    start_time_frame = time.time()
                    # full prediction on image_path
                    input_points, label_list, seg, mask_list = full_prediction(
                        img_path,
                        frame_nr,
                        merged_gaze_df,
                        sam,
                        seg_model,
                        processor,
                        presegs=True,
                        prepath=result_seg_path,
                    )
                    # if there are input points
                    if len(input_points) > 0:
                        for i in range(len(input_points)):
                            input_point, label, mask = (
                                input_points[i],
                                label_list[i],
                                mask_list[i],
                            )
                            # Append the data to the list
                            x = input_point[0]
                            y = input_point[1]
                            label_name = [item[0] for item in label]
                            overlaps = [item[1] for item in label]
                            certainty = [item[2] for item in label]

                            data_list.append(
                                [
                                    session_name,
                                    frame_nr,
                                    x,
                                    y,
                                    label_name,
                                    overlaps,
                                    certainty,
                                ]
                            )
                            # print the data
                            print(
                                session_name,
                                frame_nr,
                                x,
                                y,
                                label_name,
                                overlaps,
                                certainty,
                            )

                    else:
                        print("No fixation for frame: ", frame_nr), len(input_points)
                        data_list.append(
                            [session_name, frame_nr, None, None, None, None, None]
                        )
                    track_file.write(
                        f"{datetime.datetime.now()}:{session_name},{frame_nr},{len(input_points)},{(time.time() - start_time_frame)}\n"
                    )

                except ValueError as e:
                    print(f"A ValueError occurred: {e}")
                    print("Error with:  ", session_name, frame_nr)
                    data_list.append(
                        [session_name, frame_nr, None, None, None, None, None]
                    )

            else:
                print(f"Path does not exist: {img_path} or {seg_path}")
            # write time to a file
            print(
                f"--- {(time.time() - start_time_frame)} seconds --- for image {frame_nr}"
            )

        print(f"--- {(time.time() - start_time)} seconds --- for {n}")

        # Convert the list to a DataFrame
        df = pd.DataFrame(data_list, columns=fields)

        # Save the DataFrame to a CSV file
        csv_filename = result_label_path + n_str + "_labels_newseg.csv"
        df.to_csv(csv_filename)
        track_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process session name.")
    parser.add_argument(
        "--session", type=str, required=True, help="Session name to process."
    )
    parser.add_argument('--start_frame', type=str, required=True, help='Session name to process.')
    parser.add_argument('--end_frame', type=str, required=True, help='Session name to process.')
    args = parser.parse_args()
    
    ## if you want to run the code by just giving the session id and take the rest from the frame file, use this
    # by_frame_file = True 
    # session_file_path = folder_path + "/frame_start_end_file.csv"
    # main(args.session, " ", " ", by_frame_file, session_file_path)
    
    # otherwise use this
    main(args.session, args.atart_fram, args.end_frame)
