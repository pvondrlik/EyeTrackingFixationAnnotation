"""
File: 4.1.1_create_random_eval_set.py
Description: This script generates a list of random fixations.
Author: pvondrlik
Version: 1.0
Last Updated: 2024-03-08
"""

import pandas as pd

folder_path = "/data/"  # rename to your actual folder path
df_sessions = pd.read_csv(
    "/data/start_end_frames.csv"  # rename to your actual file path
)

# Initialize an empty DataFrame for the test set
eval_set_df = pd.DataFrame()

# Specify the columns you want to keep
columns_of_interest = [
    "session",
    "frame_nr",
    "x",
    "y",
]  # Adjust these to your actual column names

# Iterate over each unique session
for i, row in df_sessions.iterrows():
    # Filter the DataFrame for the current session
    session_name = row["session"]

    fixation_and_labels_extended = (
        folder_path + session_name + "/fixation_and_labels_extended.csv"
    )
    df = pd.read_csv(fixation_and_labels_extended)

    # Randomly select 10 rows from this session's DataFrame
    # If the session has less than 10 rows, take all rows
    df = df.sample(
        n=min(10, len(df)), random_state=42
    )  # Set a random_state for reproducibility
    print(len(df))

    # Select only the columns of interest from the sampled rows
    df = df[columns_of_interest]

    # Append the sampled rows to the test set DataFrame
    eval_set_df = pd.concat([eval_set_df, df], ignore_index=True)

# Save the test set DataFrame to a CSV file
eval_set_df.to_csv(
    "/data/cyprus_eval_frames_label.csv",  # rename to your actual file path
    index=False,
)
