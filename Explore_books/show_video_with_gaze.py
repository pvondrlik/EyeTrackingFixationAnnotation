import pandas as pd
import numpy as np
import cv2 as cv2

# file such that it does not require the librarys

def show_video(video_path, start_frame = 0, end_frame = 1000, show_gaze = False, show_label= False, merged_gaze_df = None,  merged_label_df = None):
    """
    This function shows the frames of the video with the gaze and fixation points

    Parameters:
    - video_path (str): Path to the video file.
    - start_frame (int): Start frame number.
    - end_frame (int): End frame number.
    - show_gaze (bool): Whether to show gaze points.
    - show_fix (bool): Whether to show fixation points.
    - show_label (bool): Whether to show label.
    - merged_gaze_df (pd.DataFrame): Dataframe containing gaze points.
    - merged_fix_df (pd.DataFrame): Dataframe containing fixation points.
    - merged_label_df (pd.DataFrame): Dataframe containing labels.
    
    """
    # Set start frame
    cap = cv2.VideoCapture(video_path)
   
    # get video properties
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # the cap.set is not working. therfore I use this one
    for frame in range(start_frame):  #TODO: fangen frames by 0 ode 1 an
        ret, frame = cap.read()
        if not ret:
            break


    # Loop through frames
    for frame_nr in range(start_frame, end_frame + 1):
        ret, frame = cap.read()

        if not ret:
            break  # Break the loop if video ends
        
        if show_gaze:
            filtered_df = merged_gaze_df[merged_gaze_df['frame_nr'] == (frame_nr)].reset_index(drop=True)
            print(filtered_df)
            if not filtered_df.empty:
                x = int(filtered_df.loc[0, 'gaze x [px]'])
                y = int(filtered_df.loc[0, 'gaze y [px]'])
                x, y = round(x), round(y)
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                # print all circles
                for x,y in zip(filtered_df['gaze x [px]'],filtered_df['gaze y [px]']):
                    x, y = round(x), round(y)
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)


        if show_label:
              # put box
            n = 500
            cv2.rectangle(frame, (int(frame_width/2) - 100, 45) , (int(frame_width/2) + 100, 85) , (255, 0, 0), -1)
            overlay = frame.copy()

             # Add the text to the frame
            filtered_df = merged_label_df[merged_label_df['frame_nr'] == (frame_nr)]
            label = filtered_df.loc[0, 'label']
            try :
                cv2.putText(
                    img = frame,
                    text = label,
                    org = (int(frame_width/2) - 100 + 5, 80),
                    fontFace = cv2.FONT_HERSHEY_DUPLEX,
                    fontScale = 1,
                    color = (125, 246, 55),
                    thickness = 1
                )
            except:
                pass

            cv2.addWeighted( frame, 0.5, overlay, 0.5, 0)



        # Display the frame with gaze overlay
        cv2.imshow('Video with Gaze Overlay', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release video capture object
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
    

def get_gaze_per_frame(gaze_path, world_path):
    """
    Get the gaze data from the csv file and merge it with the video frames.
    """
    oftype = {"timestamp [ns]": np.uint64}
    # load world date "section id,recording id,timestamp [ns]""
    world_path = world_path
    dfw = pd.read_csv(world_path,dtype=oftype) # load frame
    dfw['frame_nr'] = dfw.index
    dfw['timestamp [ns] world'] = dfw['timestamp [ns]']

    # load gaze data "section id,recording id,timestamp [ns],gaze x [px],gaze y [px],worn,fixation id,blink id,azimuth [deg],elevation [deg]""
    gaze_path = gaze_path
    dfg = pd.read_csv(gaze_path, dtype=oftype) # load frame7
    dfg['timestamp [ns] gaze'] = dfg['timestamp [ns]']

    # für jeden gaze einen frame
    merged_df = pd.merge_asof(
        dfw,
        dfg,
        on="timestamp [ns]", 
        direction='nearest',
        suffixes=["video", "gaze"]
    )
    return merged_df[['frame_nr','timestamp [ns]','fixation id', 'blink id',  'gaze x [px]', 'gaze y [px]']]

def get_frame_per_gaze(gaze_path, world_path):
    """
    Get the gaze data from the csv file and merge it with the video frames.
    """
    oftype = {"timestamp [ns]": np.uint64}
    # load world date "section id,recording id,timestamp [ns]""
    world_path = world_path
    dfw = pd.read_csv(world_path,dtype=oftype) # load frame
    dfw['frame_nr'] = dfw.index
    dfw['timestamp [ns] world'] = dfw['timestamp [ns]']

    # load gaze data "section id,recording id,timestamp [ns],gaze x [px],gaze y [px],worn,fixation id,blink id,azimuth [deg],elevation [deg]""
    gaze_path = gaze_path
    dfg = pd.read_csv(gaze_path, dtype=oftype) # load frame7
    dfg['timestamp [ns] gaze'] = dfg['timestamp [ns]']

    # für jeden gaze einen frame
    merged_df = pd.merge_asof(
        dfg,
        dfw,
        on="timestamp [ns]", 
        direction='nearest',
        suffixes=["video", "gaze"]
    )
    return merged_df[['frame_nr','timestamp [ns]','fixation id', 'blink id',  'gaze x [px]', 'gaze y [px]']]

def get_fix(fix_path, world_path):
    """
    Get the fixation data from the csv file and merge it with the video frames.
    """
    world_path = world_path
    dfw = pd.read_csv(world_path) # load frame

    fix_path = fix_path
    dfg = pd.read_csv(fix_path) # load frame7

    dfg['timestamp [ns]'] = dfg['end timestamp [ns]']


    dfw['frame_nr'] = dfw.index
    dfw['timestamp [ns] world'] = dfw['timestamp [ns]']
    dfg['timestamp [ns] gaze'] = dfg['timestamp [ns]']


    # für jede fixation ein frame
    merged_df = pd.merge_asof(dfw,dfg, on="timestamp [ns]", direction='forward')

    return merged_df[['frame_nr','timestamp [ns]','fixation id', 'blink id', 'timestamp [ns] gaze', 'timestamp [ns] world', 'gaze x [px]', 'gaze y [px]']]


def main():
    
    # example usage
    # gaze path 
    gaze_path = '/gaze.csv'
    world_path = '/world_timestamps.csv'
    fix_path = '/fixations.csv'
    video_path = '/_video_raw.mp4'

    # get one gaze per frame -> simpler to show
    merged_gaze_df = get_gaze_per_frame(gaze_path, world_path)
    # get all frames per gaze -> more information
    # merged_gaze_df  = get_frame_per_gaze(gaze_path, world_path)

    # show fixation
    #merged_fix_df = get_fix(fix_path, world_path)



    show_video(video_path = video_path, start_frame = 0, end_frame = 10000, show_gaze = True, show_label= False, merged_gaze_df = merged_gaze_df)#, merged_fix_df = merged_fix_df)
    
if __name__ == "__main__":
    main()