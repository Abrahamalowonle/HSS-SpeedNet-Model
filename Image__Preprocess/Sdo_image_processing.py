import glob
import cv2
import numpy as np
import sunpy.map
from astropy.io import fits 

import sunpy.visualization.colormaps as cm
from sunpy.map.header_helper import make_heliographic_header

def Image_processing(path, three_channel_save_path, binary_path):
    """
    Image Processing of SDO/AIA 193 into Standardized Three Channel EUV MAP and BINARY MAP.
    Saved as FIT Files.
    Parameters:
    ----------
    path : str
        The root directory or file path where the SDO/AIA 193 FULL disc fit Maps are located.
        e.g "/home/SDO/AIA_1200UT/*.*"
    three_channel_save_path : str
        The root directory or file path where the Standardized Three Channel EUV MAP should be stored.
        e.g "/home/Channel"
    binary_path : str
        The root directory or file path where the BINARY MAP should be stored.
        e.g "/home/Binary"
        
    Returns:
    -------
    None.
    
    Examples:
    ---------
    >>> Image_processing(path="/home/SDO/AIA_1200UT/*.*",
                         three_channel_save_path="/home/Channel",
                         binary_path="/home/Binary")
    """

    filename = sorted(glob.glob(path))
    image_index = 0
    for file in filename:
        image_index +=1
        f_name = file[-26:-15]
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
        
        # Remove NAN Effects
        non_nan_log_i = log_i[~np.isnan(log_i)]
        mini = np.min(non_nan_log_i)
        maxi = np.max(non_nan_log_i)
        stdd = np.std(non_nan_log_i)
        meann = np.mean(non_nan_log_i)
        
        std_log_image = ((log_i - meann)/stdd)

        # Three Channel
        three_channel_image = np.stack((std_log_image,)*3, axis=-1)    
        three_channel_image = three_channel_image.astype(np.float32)
        hdu = fits.PrimaryHDU(three_channel_image)
        hdul = fits.HDUList([hdu])
        # # # Write the FITS file to disk
        hdul.writeto(f"{three_channel_save_path}/{f_name}", overwrite=True)

        ### Binary Map
        resized_image_copy = resized_image.copy()
        heinemann_thresh = 0.29* np.median(resized_image_copy) + 11.53
        heinemann_binary = resized_image_copy > heinemann_thresh
        heinemann_binary_p = heinemann_binary < 1

        binary_int = heinemann_binary_p.astype(int)
        hdu_binary = fits.PrimaryHDU(binary_int)
        hdul_binary_fit = fits.HDUList([hdu_binary])

        # # # Write the FITS file to disk
        hdul_binary_fit.writeto(f"{binary_path}/{f_name}", overwrite=True)
        print(f"{f_name}: *****************{image_index /len(filename)*100:.2f}%")
