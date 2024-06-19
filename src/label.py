import numpy as np
import torch
from torch import nn
import time
from PIL import Image
import torch.nn.functional as F
from skimage.transform import resize

MIN_TO_COUNT_AS_LABEL = 0.25


# 4.2 version
# called by segment_prediction
def get_logits(img, model, preprocessor, upsample=True):
    """
    Calculate the logits for the given image and model.

    Args:
    -----
    img : np.array
        The image to predict.
    model : torch.nn.Module
        The model to use for prediction.
    preprocessor : torchvision.transforms
        The preprocessor to use for the image.
    upsample : bool
        Whether to upsample the logits to the original image size.
        Default is True.
        Set True if you want to use the logits.
        Set False if you want to store the logits.

    Returns:
    --------
    logits : torch.Tensor
        The logits matrix.

    """
    model.eval()
    # preprocess image
    pixel_values = preprocessor(img, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(model.device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    logits = outputs.logits.cpu()
    if upsample == True:
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=img.shape[:-1],  # (height, width)
            mode="bilinear",
            align_corners=False,
        )

        # Second, apply argmax on the class dimension
        logits = upsampled_logits.argmax(dim=1)[0]
    else:
        print("logits are not upsampled to original size")
    return logits


# only feasable for the notupsampled version
def load_logits(seg_path):
    """
    Load the logits from the specified path and return the label and probability matrix.

    Args:
    -----
    seg_path : str
        The path to the segmentation mask.

    Returns:
    --------
    logits : torch.Tensor
        The logits matrix.Output
    label : torch.Tensor
        The label matrix.
    prob : torch.Tensor
        The probability matrix.

    """
    # Load logits from the specified path
    logits = torch.load(seg_path)
    print(logits.shape)

    # Check if the shape matches the expected shape
    if logits.shape == torch.Size([1, 19, 128, 128]):  # size of logits from segformer
        logits = logits.float()  # Convert to 'float' data type
        logits = F.softmax(logits, dim=1)

        # Upsample the logits to match the image shape
        upsampled_logits = logits
        # Get both the maximum probabilities and their corresponding labels
        max_probs, labels = torch.max(upsampled_logits, dim=1)
        label = labels[0]  # get most probable label matrix
        prob = max_probs[
            0
        ]  # get probability matrix corresponding to the most probable labels
    else:
        raise ValueError("The shape of the logits does not match the expected shape.")
    return logits, label, prob


# newer version taking the seg prob into account
def get_labels_in_segment(seg_model, seg_label, seg_probs, mask, fq=5):
    """
    Return the counts of each label in the segmentation mask. (Step 3)

    Args:
    -----
    seg_model: segmentation model
    seg_label: segmentation label
    seg_probs: segmentation probability
    mask: mask what belongs to the object given by SAM
    fq: frequence of checked pixels

    Returns:
    --------
    label_counts : list of pixels per lable
    """

    label_counts = np.zeros(len(seg_model.config.label2id))
    label_probs = np.zeros(len(seg_model.config.label2id))
    for x in range(0, seg_label.shape[0], fq):
        for y in range(0, seg_label.shape[1], fq):
            if mask[x, y] == True:
                id = int(seg_label[int(x), int(y)])
                if seg_probs[int(x), int(y)] > 0.2:
                    label_counts[id] += 1
                    label_probs[id] += seg_probs[int(x), int(y)]

    label_probs = label_probs / label_counts
    return label_counts, label_probs


def find_label_from_counts(counts, probs):
    """
    Return the labels, the certainty of the label and the percentage of the label in the mask.

    Args:
    -----
    counts: list of counts per label
    probs: list of probabilities per label

    Returns:
    --------
    labels : list of labels
    seg_certaintys : list of certainties
    percents : list of percentages
    """

    suml = sum(counts)
    labels = []  # label
    percents = []  # percent of that label of the whole mask
    seg_certaintys = []  # the summed certainty of the label

    for i, c in enumerate(counts):
        if c >= MIN_TO_COUNT_AS_LABEL * suml:
            labels.append(i)
            percents.append(c / suml)
            seg_certaintys.append(probs[i])

    return labels, seg_certaintys, percents


# ultralytics version full prediciton on image path
# calls get_logits, load_logits, get_labels_in_segment, find_label_from_counts
def full_prediction(
    img_path,
    frame_nr,
    merged_gaze_df,
    sam,
    seg_model,
    processor,
    presegs=False,
    prepath=None,
):
    """
    Args:
    -----
    - img_path: str
            image to predict
    - frame_nr: int
            frame number to find the correct gaze values
    - merged_gaze_df: pandas dataframe
            dataframe containing gaze coordinates per frame
    - sam: model
            sam modell to predict the mask for the object
    - seg_model: model
            to predict the segmentation
    - processor:

    Returns:
    --------
    - input_points: list of tuples
            gaze points
    - label_list: list of lists
            list of labels
    - seg_label: matrix
            segmentation mask
    - masks_list: list of matrices
            list of masks
    """
    img = Image.open(img_path)
    img_array = np.array(img)

    filtered_df = merged_gaze_df[merged_gaze_df["frame_nr"] == frame_nr].reset_index(
        drop=True
    )
    # for every point
    input_points = []
    for i in filtered_df.index:
        input_points.append(
            (
                int(filtered_df["gaze x [px]"].iloc[i]),
                int(filtered_df["gaze y [px]"].iloc[i]),
            )
        )

    # if there is no fixation for that frame
    label_list, masks_list = [], []
    if len(input_points) > 0:

        if presegs:
            frame_nr_str = f"{frame_nr :05d}"
            seg_path = prepath + "frame_" + frame_nr_str + ".pt"
            print("Load Segmask: ", seg_path)
            logits, seg_label, seg_prob = load_logits(seg_path)

        else:
            logits, seg_label, seg_prob = get_logits(img_array, seg_model, processor)

        # get sam
        input_label = [1] * len(input_points)
        start_block = time.time()
        results = sam(img_path, points=input_points, labels=input_label)  # get n masks
        print(f"Time to get sam: {time.time() - start_block} seconds")

        start_block = time.time()
        for mask in results[0].masks.data:
            # get masks
            masks_list.append(mask)
            downsampled_mask = resize(mask, (128, 128), anti_aliasing=False)

            # get label matrix by comparing seg with mask
            label_counts, label_probs = get_labels_in_segment(
                seg_model, seg_label, seg_prob, downsampled_mask, fq=1
            )
            labels, seg_certaintys, percents = find_label_from_counts(
                label_counts, label_probs
            )
            label = [
                (seg_model.config.id2label[l], round(p, 5), round(c, 5))
                for l, p, c in zip(labels, percents, seg_certaintys)
            ]
            label_list.append(label)

        print(f"Time to calc labels: {time.time() - start_block} seconds")

    else:
        seg_label = None
    return input_points, label_list, seg_label, masks_list


# calls  get_logits
def segment_prediction(img_path, frame_nr, seg_model, processor):
    """
    Args:
    -----
    - img_path: str
            image to predict
    - frame_nr: int
            frame number to find the correct gaze valuesS
    - seg_model: model
            to predict the segmentation
    - processor:

    Returns:
    -------
    - logits : matrix
            segmentation mask

    """

    img = Image.open(img_path)
    img_array = np.array(img)
    # segmentation
    logits = get_logits(img_array, seg_model, processor, upsample=False)
    return logits
