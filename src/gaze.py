import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

# used in step 1-3 (on HPC)
def get_frame_per_gaze(gaze_path, world_path):
    """
    Get the gaze data from the csv file and merge it with the video frames.
    """
    oftype = {"timestamp [ns]": np.uint64}
    # load world date "section id,recording id,timestamp [ns]""
    world_path = world_path
    dfw = pd.read_csv(world_path, dtype=oftype)  # load frame
    dfw["frame_nr"] = dfw.index
    dfw["timestamp [ns] world"] = dfw["timestamp [ns]"]

    # load gaze data "section id,recording id,timestamp [ns],gaze x [px],gaze y [px],worn,fixation id,blink id,azimuth [deg],elevation [deg]""
    gaze_path = gaze_path
    dfg = pd.read_csv(gaze_path, dtype=oftype)  # load frame
    dfg["timestamp [ns] gaze"] = dfg["timestamp [ns]"]

    # for each gaze one frame
    merged_df = pd.merge_asof(
        dfg, dfw, on="timestamp [ns]", direction="nearest", suffixes=["video", "gaze"]
    )
    return merged_df[
        [
            "frame_nr",
            "timestamp [ns]",
            "fixation id",
            "blink id",
            "gaze x [px]",
            "gaze y [px]",
        ]
    ]


#
