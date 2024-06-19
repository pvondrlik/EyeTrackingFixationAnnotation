import pandas as pd
import os

# change path that contains information about start and end frames
loacation_path = ""
path_sessions = loacation_path + "/start_end_frames.csv"
df = pd.read_csv(path_sessions)
idx = 3

session = df["session"][idx]
start = df["start"][idx]
end = df["end"][idx]
print(session)


folder_path = loacation_path + session + "/Labels/"
output_file = loacation_path + session + "/all_labels_newseg.csv"
image_path = loacation_path + session + "/video_frames_img/"
seg_path = loacation_path + session + "/Segmasks/"
dfs = []

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith("newseg.csv"):  # chek only files belonging to a certain run
        # Construct full file path
        file_path = os.path.join(folder_path, filename)
        # Read the CSV file and append to list
        df = pd.read_csv(file_path)
        dfs.append(df)

# Concatenate all DataFrames in the list
combined_df = pd.concat(dfs, ignore_index=True)

# Sort by 'frame_nr' column
sorted_df = combined_df.sort_values(by="frame_nr", ignore_index=True)

# Reset index
sorted_df.reset_index(drop=True, inplace=True)  # .drop_duplicates()

# set contatinin all frame numbers
actual_set = set(sorted_df["frame_nr"])

# print which numbers are missing
expected_set = set(range(start, end))

missing_numbers = expected_set - actual_set
print(f"Missing numbers: {sorted(missing_numbers)}")

for number in missing_numbers:
    nr = f"{number :05d}"
    if not os.path.exists(image_path + "frame_" + nr + ".jpg"):
        print(f"Missing Image {number}")
    if not os.path.exists(seg_path + "frame_" + nr + ".pt"):
        print(f"Missing Segmask {number}")

print(session, len(missing_numbers))

try:
    sorted_df.drop(columns=["Unnamed: 0"], inplace=True)
except:
    pass

# Write the sorted DataFrame to a new CSV file
sorted_df.to_csv(output_file, index=False)
