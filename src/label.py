import numpy as np
import torch
from torch import nn
from PIL import Image
import time
import torch.nn.functional as F

MIN_TO_COUNT_AS_LABEL = 0.25
# newer cippy verion
def get_logits_cippy(img, model, preprocessor, upsample = True):
    model.eval()
    # preprocess image
    pixel_values = preprocessor(img, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(model.device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    logits = outputs.logits.cpu()
    if upsample == True:
        upsampled_logits = nn.functional.interpolate(logits,
                        size=img.shape[:-1], # (height, width)
                        mode='bilinear',
                        align_corners=False)

        # Second, apply argmax on the class dimension
        seg = upsampled_logits.argmax(dim=1)[0]
    else :
        print("logits are not upsampled to original size")
        seg = logits
    return seg

# 4.2 version
def get_logits_older_version(img, model, preprocessor):
    
    model.eval()
    # preprocess image
    pixel_values = preprocessor(img, return_tensors="pt").pixel_values

    pixel_values = pixel_values.to(model.device)



       # Get the logits from the model
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    logits = outputs.logits.cpu()
    # the output is of size 265 always so it has to be upsampled
    upsampled_logits = nn.functional.interpolate(logits,
                    size=img.shape[:-1], # (height, width)
                    mode='bilinear',
                    align_corners=False)

    # Second, apply argmax on the class dimension
    seg = upsampled_logits.argmax(dim=1)[0]
    return seg

# new version beause its taking the softmax and requires the not upsampled logits... merge with othere function
def load_logits( seg_path):
    # Load logits from the specified path
    logits = torch.load(seg_path)   
    print(logits.shape) 

    # Check if the shape matches the expected shape
    if logits.shape == torch.Size([1, 19, 128, 128]):
        logits = logits.float()  # Convert to 'float' data type
        logits = F.softmax(logits, dim=1)
        
        # Upsample the logits to match the image shape
        upsampled_logits = logits #nn.functional.interpolate(logits, size=img_shape, mode='bilinear', align_corners=False)
        # Get both the maximum probabilities and their corresponding labels
        max_probs, labels = torch.max(upsampled_logits, dim=1)
        label = labels[0]  # Extract the most probable label matrix
        prob = max_probs[0]  # Extract the probability matrix corresponding to the most probable labels
    
        return logits,label, prob  # Return both the label matrix and the probability matrix


# functions to find label
def get_labels_in_segment(seg_model,seg,mask, fq = 5): # TODO just give the length 
    """
    input:
    - seg_model: segmentation model
    - seg: segmentatioin by segformer
    - masks: mask what belongs to the object given by SAM
    - fq: frequence of checked pixels
    output:
    - lable_counts : list of pixels per lable
    """
    label_counts = np.zeros(len(seg_model.config.label2id))

    for x in range(0,seg.shape[0], fq):
        for y in range(0,seg.shape[1], fq):
            if mask[x,y] == True:
                id = int(seg[int(x), int(y)])
                label_counts[id] += 1
    return label_counts

# newer version taking the seg prob into account
def get_labels_in_segment(seg_model,seg_label,seg_probs,mask, fq = 5): # TODO just give the length 
    """
    input:
    - seg_model: segmentation model
    - seg: segmentatioin by segformer
    - masks: mask what belongs to the object given by SAM
    - fq: frequence of checked pixels
    output:
    - lable_counts : list of pixels per lable
    """
    
    label_counts = np.zeros(len(seg_model.config.label2id))
    label_probs = np.zeros(len(seg_model.config.label2id))
    for x in range(0,seg_label.shape[0], fq):
        for y in range(0,seg_label.shape[1], fq):
            if mask[x,y] == True:
                id = int(seg_label[int(x), int(y)])
                if seg_probs[int(x), int(y)] > 0.2 : 
                    label_counts[id] += 1
                    label_probs[id] += seg_probs[int(x), int(y)]
                    

    label_probs = label_probs / label_counts
    return label_counts , label_probs



# get the label if higher than...
def find_label_from_counts(counts):
    """
    input:
    - counts: list of counts per label
    output:
    - maxs: the ammount of pixles
    - ps: position of lable
    """
    suml = sum(counts)
    maxs = []
    ps = []
    for i, c in enumerate(counts):
        if c >= MIN_TO_COUNT_AS_LABEL * suml:
            maxs.append(i)
            ps.append(c/suml)
    return maxs, ps


#newer version taking the seg prob into account
def find_label_from_counts(counts, probs):
    """
    input:
    - counts: list of counts per label
    output:
    - maxs: the ammount of pixles
    - ps: position of lable
    """

    suml = sum(counts)
    labels = [] # label
    percents = [] # percent of that label of the whole mask
    seg_certaintys = [] # the summed certainty of the label


    for i, c in enumerate(counts):
        if c >= MIN_TO_COUNT_AS_LABEL * suml:
            labels.append(i)
            percents.append(c/suml)
            seg_certaintys.append(probs[i])
    
    return labels, seg_certaintys, percents

#  ultralytics version full prediciton on image path 
def full_prediction(img_path, frame_nr, merged_gaze_df, sam, seg_model, processor):
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

    Output:
    -------
    - frame_nr: int
    - input_point: tuple
    - label: str

    """
    # extract the image
    img = Image.open(img_path)
    img_array = np.array(img)

    # get fixation
    filtered_df = merged_gaze_df[merged_gaze_df['frame_nr'] == frame_nr].reset_index(drop=True)
    # for verx pint
    input_points = []
    for i in filtered_df.index:
        input_points.append((int(filtered_df['gaze x [px]'].iloc[i]), int(filtered_df['gaze y [px]'].iloc[i])))

    # if there is no fixation for that frame
    label_list, masks_list = [], []
    if len(input_points) > 0:

       # get segmentation (Segformer)
        start_block = time.time()
        seg = get_logits(img_array, seg_model, processor)
        print(f"Time to get segmentation (Segformer): {time.time() - start_block} seconds")

        # get sam
        input_label = [1]*len(input_points)
        start_block = time.time()
        results = sam(img_path, points=input_points, labels=input_label) # get n masks
        print(f"Time to get sam: {time.time() - start_block} seconds")

        start_block = time.time()
        for mask in results[0].masks.data:
            # get masks
            masks_list.append(mask)
            # get label matrix by comparing seg with mask
            l2 = get_labels_in_segment(seg_model, seg, mask)
            lfc = find_label_from_counts(l2)
            label = [(seg_model.config.id2label[x], round(p, 3)) for x, p in zip(lfc[0], lfc[1])]
            label_list.append(label)
        print(f"Time to calc labels: {time.time() - start_block} seconds")
    
    return  input_points, label_list, seg, masks_list


def full_predictionold(img_path, frame_nr, merged_gaze_df, sam, seg_model, processor):
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

    Output:
    -------
    - frame_nr: int
    - input_point: tuple
    - label: str

    """
    img = Image.open(img_path)
    img_array = np.array(img)

    # get fixation
    filtered_df = merged_gaze_df[merged_gaze_df['frame_nr'] == frame_nr].reset_index(drop=True)

    # for verx pint
    input_points = []
    for i in filtered_df.index:
        input_points.append((int(filtered_df['gaze x [px]'].iloc[i]), int(filtered_df['gaze y [px]'].iloc[i])))


    # if there is no fixation for that frame
    label_list, masks_list = [], []

    if len(input_points) > 0:
       # get segmentation (Segformer)
        start_block = time.time()
        seg = get_logits(img_array, seg_model, processor)
        print(f"Time to get segmentation (Segformer): {time.time() - start_block} seconds")

        for input_point in input_points:

            start_block = time.time()
            input_label = np.array(1)


            # get mask (SAM)
            start_block = time.time()
            results = sam(img_path, points=input_point, labels=input_label)
            masks, scores = results[0].masks, results[0].probs
            masks_list.append(masks.data)
            end_block = time.time()
            print(f"Time to get SAM: {end_block - start_block} seconds")

            # get label matrix by comparing seg with mask
            l2 = get_labels_in_segment(seg_model, seg, masks.data)
            lfc = find_label_from_counts(l2)
            print(lfc, lfc[0], lfc[1])
            label = [(seg_model.config.id2label[x], round(p, 3)) for x, p in zip(lfc[0], lfc[1])]
            print(label)
            label_list.append(label)

            # Total duration
            #print(f"Time to get label matrix: {end_block - start_block} seconds")
            #total_duration = time.time() - start_time
            #print(f"Total duration of the function: {total_duration} seconds")
        
    print("label:",frame_nr ,input_points,label_list)
    return  input_points, label_list, seg, masks_list


def segment_prediction(img_path, frame_nr,seg_model, processor):
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

    Output:
    -------
    - logits : matrix
            segmentation mask

    """
    start_block = time.time()

    img = Image.open(img_path)
    img_array = np.array(img)
    # segmentation


    # set upsample to FALSE 
    logits = get_logits_cippy(img_array, seg_model, processor, upsample = True)

    #print(f"Time to get seg for {frame_nr} : {time.time() - start_block} seconds")

    return  logits