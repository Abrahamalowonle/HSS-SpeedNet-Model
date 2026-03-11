import glob
import cv2
import numpy as np
import sunpy.map
from astropy.io import fits 

import sunpy.visualization.colormaps as cm
from sunpy.map.header_helper import make_heliographic_header

def Image_processing(file, meann, stdd, binary_operate=False):
    """
    Image Processing of SDO/AIA 193 into Standardized Three Channel EUV MAP and BINARY MAP.
    Saved as FIT Files.
    Parameters:
    ----------
    file : str
        The root directory or file path where the SDO/AIA 193 FULL disc fit Maps are located.
        e.g "./DATASET/AIA/171/{year}/*.*"
    mean : float
        The Average of the Mean value of Each Channel (171, 193, or 304)
    stdd : float
        The Average of the STD value of Each Channel (171, 193, or 304)
    binary_operate : bool
        If False, do not obtain the binary map, Else True (For AIA 193, obtain the binary map).
        
    Returns:
    -------
    standardized_projected_304, and Binary_map.
    
    Examples:
    ---------
    >>> Image_processing(path="./DATASET/AIA/171/{year}/*.*",
                         meann, stdd, binary_operate=False")
    """
    aia_map = sunpy.map.Map(file)
    shape = (1440, 2880)
    carr_header = make_heliographic_header(aia_map.date,
                                            aia_map.observer_coordinate,
                                            shape, frame='stonyhurst',)
    outmap = aia_map.reproject_to(carr_header)
    stonyhurst_reprojection = outmap.data
    flip = stonyhurst_reprojection[::-1,:]

    nan_mask = np.isnan(flip)
    
    center_row = flip.shape[0] // 2

    center_non_nan_cols = np.where(~nan_mask[center_row])[0]
    
    if len(center_non_nan_cols) > 0:
        start_col = center_non_nan_cols[0]
        end_col = center_non_nan_cols[-1] + 1
    
        # Find the first and last non-NaN in the columns of the bounding box
        rows_with_nan_in_box = np.any(~nan_mask[:, start_col:end_col], axis=1)
        start_row = np.argmax(rows_with_nan_in_box)
        end_row = len(rows_with_nan_in_box) - np.argmax(rows_with_nan_in_box[::-1])
    
        cropped_image = flip[start_row:end_row, start_col:end_col]
    else:
        cropped_image = np.array([])  
    
    nan_mask = np.isnan(cropped_image)
    cropped_image[nan_mask] = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), cropped_image[~nan_mask])

    new_shape = 256
    if cropped_image.shape[0] > 0:
        resized_image = cv2.resize(cropped_image, (new_shape, new_shape), interpolation=cv2.INTER_LINEAR)
    else:
        resized_image = np.full((new_shape, new_shape), np.nan)
    
    # Standardizing the image from the logrithmic scale
    log_i = np.log(resized_image)
    
    
    std_log_image = ((log_i - meann)/stdd)

    resized_image_copy = resized_image.copy()
    binary_p = np.zeros_like(resized_image_copy)
    
    if binary_operate:
        ### Binary Map
        thresh = 0.29* np.median(resized_image_copy) + 11.53
        binary = resized_image_copy > thresh
        binary_p = binary < 1
        
    return std_log_image, binary_p


       
