import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

label_colors = {
    0: [255, 128, 0],  # Black for road
    1: [255, 0, 0],  # Red for label 1
    2: [0, 255, 0],  # Green for label 2
    3: [0, 0, 255],  # Blue for label 3
    4: [255, 255, 0],  # Yellow for label 4
    5: [255, 0, 255],  # Magenta for label 5
    6: [0, 255, 255],  # Cyan for label 6
    7: [128, 0, 0],  # Maroon for label 7
    8: [0, 128, 0],  # Green for label 8
    9: [0, 0, 128],  # Navy for label 9
    10: [128, 128, 0],  # Olive for label 10
    11: [128, 0, 128],  # Purple for label 11
    12: [0, 128, 128],  # Teal for label 12
    13: [192, 192, 192],  # Silver for label 13
    14: [128, 128, 128],  # Gray for label 14
    15: [0, 0, 0],  # Orange for label 15
    16: [0, 255, 128],  # Lime for label 16
    17: [128, 0, 255],  # Fuchsia for label 17
    18: [128, 255, 0],  # Lime for label 19
    # You can add more colors for additional labels as needed
}


def get_label_colors():
    return label_colors

def get_seg_img(img, seg):
    """
    This function shows the image with the segmentation
    same ase show_img

    Parameters:
    - img: image
    - seg: segmentation

    Returns:
    - img: image with segmentation
    - color_seg: just segmentation
    """
    # @TODO Rename to get_seg_img

    color_seg = np.zeros(
        (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
    )  # height, width, 3\
    for label, color in label_colors.items():
        color_seg[seg == label] = color
    # Convert to BGR
    # color_seg = color_seg[..., ::-1] # kehrt die Reihenfolge der Farbkanäle um.

    # Show image + mask
    img = np.array(img) * 0.5 + color_seg * 0.5
    img_with_seg = img.astype(np.uint8)

    return img_with_seg, color_seg


def show_frames(
    video_path,
    start_frame=2000,
    n_show=25,
    fq_show=100,
    show_gaze=False,
    merged_gaze_df=None,
):
    """
    This function shows the frames of the video with the gaze and fixation points

    Parameters:
    - video_path (str): Path to the video file.
    - start_frame (int): Start frame number.
    - n_show (int): Number of frames to show.
    - fq_show (int): Frequency of frames to show.
    - show_gaze (bool): Whether to show gaze points.
    - merged_gaze_df (pd.DataFrame): Dataframe containing gaze points.

    Returns:
    - None
    """

    # plot fixation without the segmentation
    fig, axs = plt.subplots(int(sqrt(n_show)), int(sqrt(n_show)), figsize=(30, 30))
    axs = axs.flatten()

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    img_idx = 0

    start_time = time.time()

    # funktioniert eventuell nicht richtig
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_nr in range(start_frame, start_frame + n_show * fq_show):
        ret, img = cap.read()
        if ret == False or img_idx >= n_show:
            break
        if frame_nr % fq_show == 0:

            axs[img_idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[img_idx].set_title(f"Frame: {frame_nr}")
            axs[img_idx].axis("off")

            if show_gaze:
                filtered_df = merged_gaze_df[merged_gaze_df["frame_nr"] == (frame_nr)]
                for x, y in zip(filtered_df["gaze x [px]"], filtered_df["gaze y [px]"]):
                    x, y = round(x), round(y)
                    axs[img_idx].scatter(y, x, s=100, c="blue", marker="o")

            img_idx += 1

    end_time = time.time()

    print(f"total time {end_time - start_time} s")

    plt.tight_layout()
    plt.show()
    cap.release()


# working
def show_video(
    video_path,
    start_frame=0,
    end_frame=1000,
    show_gaze=False,
    show_fix=False,
    show_label=False,
    merged_gaze_df=None,
    merged_fix_df=None,
    merged_label_df=None,
):
    """
    This function shows the frames of the video with the gaze and fixation points as a video

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
    for frame in range(start_frame):  # TODO: fangen frames by 0 ode 1 an
        ret, frame = cap.read()
        if not ret:
            break

    # Loop through frames
    for frame_nr in range(start_frame, end_frame + 1):
        ret, frame = cap.read()

        if not ret:
            break  # Break the loop if video ends

        if show_gaze:
            filtered_df = merged_gaze_df[merged_gaze_df["frame_nr"] == (frame_nr)]
            if not filtered_df.empty:
                x = int(filtered_df.loc[0, "gaze x [px]"])
                y = int(filtered_df.loc[0, "gaze y [px]"])
                x, y = round(x), round(y)
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                # print all circles
                for x, y in zip(filtered_df["gaze x [px]"], filtered_df["gaze y [px]"]):
                    x, y = round(x), round(y)
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        if show_fix:
            filtered_df = merged_fix_df[merged_fix_df["frame_nr"] == (frame_nr)]
            # adding the gaze dots
            if not filtered_df.empty:
                x = int(filtered_df.loc[0, "gaze x [px]"])
                y = int(filtered_df.loc[0, "gaze y [px]"])
                x, y = round(x), round(y)
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

        if show_label:
            # put box
            n = 500
            cv2.rectangle(
                frame,
                (int(frame_width / 2) - 100, 45),
                (int(frame_width / 2) + 100, 85),
                (255, 0, 0),
                -1,
            )
            overlay = frame.copy()

            # Add the text to the frame
            filtered_df = merged_label_df[merged_label_df["frame_nr"] == (frame_nr)]
            label = filtered_df.loc[0, "label"]
            try:
                cv2.putText(
                    img=frame,
                    text=label,
                    org=(int(frame_width / 2) - 100 + 5, 80),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1,
                    color=(125, 246, 55),
                    thickness=1,
                )
            except:
                pass

            cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

        # Display the frame with gaze overlay
        cv2.imshow("Video with Gaze Overlay", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    # Release video capture object
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()


# called internaly in show_frames
def show_mask(mask, ax, color=None):
    """
    This function shows the mask on the axis.

    Args:
    - mask: Mask to be shown.
    - ax: Axis to plot the mask.
    - color: Color of the mask.
    """

    if isinstance(mask, memoryview):
        mask = np.array(mask)

    # If color is not specified, use green with 60% opacity
    if color is None:
        color = np.array([0, 1, 0, 0.6])  # RGBA

    # Assuming the mask is binary (0 or 1), we can use it to select color
    mask_image = np.zeros(
        (*mask.shape, 4)
    )  # Create an RGBA image based on the mask size
    mask_image[mask == 1] = color  # Apply color where mask is 1

    # Overlay the colored mask on the provided axis
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


# Function to print the image with the marked point
def print_image_with_point(ax, img_path, gaze_x, gaze_y, frame_nr=None, color="red"):
    """
    This function prints the image with the marked point.

    Args:
    - ax: Axis to plot the image.
    - img_path: Path to the image.
    - gaze_x: x-coordinate of the gaze point.
    - gaze_y: y-coordinate of the gaze point.
    - frame_nr: Frame number.
    - color: Color of the marked point.
    """

    # Load the image using OpenCV
    img = mpimg.imread(img_path)

    # Display the image
    ax.imshow(img)
    # Mark the point on the image
    try:
        ax.scatter(gaze_x, gaze_y, color=color, marker="x")
    except:
        for x, y in zip(gaze_x, gaze_y):
            ax.scatter(x, y, color=color, marker="*")

    # Add title
    if frame_nr is not None:
        ax.set_title(f"Frame {frame_nr}")
    ax.axis("off")


def save_single_image(mask, input_point, img_with_seg, output_filename):
    """
    Save a single image with masks, input points, and segmentation.

    Parameters:
    - masks (list): List of segmentation masks.
    - input_point (list): List of input points.
    - input_label (int): Input label.
    - img_with_seg: Image with segmentation.
    - output_filename (str): Output filename for the saved image.
    """

    # Assuming show_mask and show_points functions are defined somewhere in your code

    fig, ax = plt.subplots(figsize=(5, 5))  # Create a single subplot

    # Assuming you want to display only the first mask in the list
    show_mask(mask, ax)
    ax.scatter(
        input_point[0],
        input_point[1],
        color="green",
        marker="*",
        edgecolor="white",
        linewidth=1.25,
    )

    ax.imshow(img_with_seg)
    ax.axis("off")  # Turn off x and y axes

    # Save the figure as an image
    plt.savefig(output_filename, bbox_inches="tight", pad_inches=0)
    plt.close()  # Close the figure to release resources


def show_sam_seg_img(image, point, seg_mask, sam_mask, title="Segmentation Mask"):
    """
    This function shows the image with the segmentation
    same ase show_img

    Parameters:
    - img: image
    - seg: segmentation

    Output:
    - image with original image, segmentation and combination of both
    """
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    for i in range(4):
        axs[i].scatter(
            point[0], point[1], color="red", marker="x", edgecolor="red", linewidth=1.25
        )

    # Original image with points (assuming points are in the mask)
    axs[0].imshow(image)
    axs[0].set_title("Original Image")

    # Just the mask
    show_mask(sam_mask, axs[1])
    axs[1].imshow(sam_mask, cmap="gray")
    axs[1].set_title("Mask")

    segimg, _ = get_seg_img(image, seg_mask)

    # Segmentation mask
    axs[2].imshow(segimg, cmap="gray")
    axs[2].set_title("Segmentation Mask")

    axs[3].imshow(segimg, cmap="gray")
    axs[3].set_title("Segmentation with Mask")
    show_mask(sam_mask, axs[3])

    # set title
    fig.suptitle(title)

    for ax in axs:
        ax.axis("off")
    plt.show()


# used in 4.2.2 and 4.3.2
def show_complete_fixation_with_all_frames_all_gaze(
    df, fix_nr, img_path, labels=False, max_img=False
):
    """
    This function can visualize a whole fixation with all frames and gaze points.

    Args:
    -----
    - df: DataFrame with the fixation data.
    - fix_nr: Fixation number.
    - img_path: Path to the images.
    - labels: Whether to show labels.
    - max_img: Whether to show all images or just a few.

    Ourput:
    -------
    - Visualization of the fixation.

    """
    # get the fixation
    selection = df.groupby(["fixation_id"]).get_group((fix_nr))

    # get the frames in fixation
    num_frames = selection["frame_nr"].nunique()

    num_cols = 3
    WIDTH = 3
    if max_img == False:
        num_rows = int(np.ceil(num_frames / num_cols))
        fig, axs = plt.subplots(
            num_rows, num_cols, figsize=(num_cols * WIDTH * 1.3, num_rows * WIDTH)
        )
    else:
        num_rows = 2  # now its limited to 6 images
        fig, axs = plt.subplots(
            num_rows, num_cols, figsize=(num_cols * WIDTH * 1.3, num_rows * WIDTH)
        )

    for i, (frame_nr, group) in enumerate(selection.groupby("frame_nr")):
        ax = axs.flat[i]
        str_frame_nr = f"{frame_nr:05d}"
        path = img_path + "frame_" + str_frame_nr + ".jpg"  # path to the image

        # list of all x and y coordinates
        x = list(group["x"])
        y = list(group["y"])
        print_image_with_point(ax, path, x, y, str(frame_nr) + ": ")
        if labels:
            label = list(group["label"])[0]
            print_image_with_point(ax, path, x, y, str(frame_nr) + ": " + str(label))

        # axis off
        ax.axis("off")
        if i == num_rows * num_cols - 1:
            break

    fig.suptitle(f"Fixation: {fix_nr}")
    plt.tight_layout()
    plt.show()
