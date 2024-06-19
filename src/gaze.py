import pandas as pd
import numpy as np
import bisect

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
    Get the gaze data from the csv file and merge it with the video frames.
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

def get_fix_path_df(n_frames, fq_show,  fps):
    """
    Get the fix_path data from the csv file and merge it with the video frames
    """
    # receive fix_path data
    fix_path = "/home/pvondrlik/Desktop/BA_Thesis/repo-movie-analysis/data/Expl_1_ET_1_2023-09-05_11-56-16_ET/fix_path.csv"
    df = pd.read_csv(fix_path) # load frame
    df['time_abs'] = df['timestamp [ns]']- df.loc[0,'timestamp [ns]']  # replace namimg and get absolut time
    
    # calculate time for each frame
    n_frames = n_frames +1
    times = [int(frame/fps *(10**9)) for frame in range(n_frames)] # calculate time for each frame
    frames = [n for n in range(n_frames)] # create list of frames
    times_df = pd.DataFrame(list(zip(times, frames)), columns=['vid_times', 'frames']) 
    
    # Find the closest lower number from list1 for each element in list
    df['vid_times'] = df['time_abs'].apply(lambda x: times[bisect.bisect_right(times, x) - 1] if x >= times[0] else np.nan)

    # merge the dataframes to get a list of fixation ids for each frame
    merged_df = pd.merge(times_df, df, on="vid_times",how='outer') 

    # drop rows that correspond to further frames - this step might be obsolete 
    t = times_df.loc[n_frames -1, 'vid_times']#
    merged_df = merged_df.drop(merged_df[merged_df['vid_times'] == t].index) # drop row that correspont do further frames 

    # drop rows that are not fixations
    merged_df = merged_df.dropna(subset=['fixation id'])
    
    return merged_df[[ 'frames','vid_times','time_abs', 'fix_path x [px]', 'fix_path y [px]', 'fixation id']] # show only necessary data

