import pandas as pd
import numpy as np


from astropy.io import fits
import random
import glob 

import datetime
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, BatchNormalization, Dense, Flatten, Conv2D, MaxPooling2D, Dense, LSTM



def load_df(path, train_start_date, train_end_date, test_start_date, test_end_date):

    """
    Processes the Filtered Solar Wind velocity and deviation from the specified path within given training and testing time ranges.

    Parameters:
    ----------
    path : str
        The root directory or file path where the Filtered SW, standard deviation of solar wind, ICME_event files are located.
    train_start_date : str
        The start time of the training period, formatted as a string (e.g., "YYYY-MM-DD").
    train_end_date : str
        The end time of the training period, formatted as a string (e.g., "YYYY-MM-DD").
    test_start_date : str
        The start time of the testing period, formatted as a string (e.g., "YYYY-MM-DD").
    test_end_date : str
        The end time of the testing period, formatted as a string (e.g., "YYYY-MM-DD").

    Returns:
    -------
    df (SW dataframe) : pandas.DataFrame,
    velocity_train : list of pandas.DataFrame, 
    velocity_test : list of pandas.DataFrame, 
    std_velocity_test : list of pandas.DataFrame , 
    train_speed_dates : list, 
    test_speed_dates : list, 
    full_icme_list : list

    Notes:
    ------
    - Ensure the time strings are correctly formatted for proper time comparisons.
    - This function assumes the files in the path include timestamps or metadata to filter based on time.
    
    Examples:
    ---------
    >>> load_df(
            path="/data/files",
            train_start_date="2010-05-13",
            train_end_date="2011-07-06",
            test_start_date="2011-07-07",
            test_end_date="2011-12-31"
        )
    """
    df = pd.read_csv(f'{path}/Omni_velocity.csv')
    
    velocity_train_mask= (df["DATE"] >= f"{train_start_date}") & (df["DATE"] <= f"{train_end_date}") 
    velocity_train = df.loc[velocity_train_mask]
   
    velocity_test_mask = (df["DATE"] >= f"{test_start_date}") & (df["DATE"] <= f"{test_end_date}")
    velocity_test= df.loc[velocity_test_mask]
    
    STD_VELOCITY = pd.read_csv(f'{path}/standard_deviation_Velocity.csv')
    STD_VELOCITY["DATE"] = pd.to_datetime(STD_VELOCITY["DATE"], format="mixed")
    std_velocity_test_mask = (STD_VELOCITY["DATE"] >= f"{test_start_date}") & (STD_VELOCITY["DATE"] <= f"{test_end_date}") 
    std_velocity_test= STD_VELOCITY.loc[std_velocity_test_mask]
    
    #Separate dates for future plotting
    train_speed_dates = list(pd.to_datetime(velocity_train["DATE"]))
    test_speed_dates = list(pd.to_datetime(velocity_test["DATE"]))
    
    icme_list = pd.read_excel(f"{path}/Original_Full_Richardson_cane_list(2010-2023).xlsx")
    
    # Loop through the start and end time
    icme_list['Start_time'] = pd.to_datetime(icme_list['Start_time'].str.split(' ').str[0], format='%Y/%m/%d')
    icme_list['End_time'] = pd.to_datetime(icme_list['End_time'].str.split(' ').str[0], format='%Y/%m/%d')
    
    start_time_list = list(icme_list['Start_time'])
    end_time_list = list(icme_list['End_time'])

    whole_list = []
    for i in range(len(start_time_list)):
        new_list = pd.date_range(start_time_list [i], end_time_list[i],) #freq='h'
        whole_list.append(new_list)
        
    full_icme_list = [date for sublist in whole_list for date in sublist]
    
    return df, velocity_train, velocity_test, std_velocity_test, train_speed_dates, test_speed_dates, full_icme_list

def non_icme_Image_date(img_train_path, img_test_path, full_icme_list):
    """
    Filters ICME events from the Maps in the specified path within given training and testing time ranges.

    Parameters:
    ----------
    img_train_path : str
        The root directory or file path where the Map training set are located.
    img_test_path : 
        The root directory or file path where the Map testing set are located.
    full_icme_list : list
        A list of ICME events to be excluded or removed from processing.
    Returns:
    -------
   Filtered Train Map --> map_train_non_icme_date : list,
   Filtered Test Map  --> map_test_non_icme_date : list
    
    Examples:
    ---------
    >>> non_icme_Image_date(
            img_train_path="/data/train_files",
            img_test_path="/data/test_files", 
            full_icme_list=full_icme_list,
        )
    """
    
    train_filename = sorted(glob.glob(f"{img_train_path}/**"))
    
    map_train_non_icme_date = []
  
    for train_file in train_filename:
        #  Extract the Date of the Image
        f_name = train_file[-8:]
        
        # Convert the String Date to Datetime Format
        solar_date_train = datetime.strptime(f_name,'%Y%m%d')
        
        # REMOVE THE ICME EVENTS
        if solar_date_train not in full_icme_list:
            # THEN APPEND THE DATES INTO A LIST
            map_train_non_icme_date.append(solar_date_train)
            
    test_filename = sorted(glob.glob(f"{img_test_path}/**"))
    
    map_test_non_icme_date = []
    
    for test_file in test_filename:
    
        #  Extract the Date of the Image 
        f_test_name = test_file[-8:]
        
        # Convert the String Date to Datetime Format
        solar_date_test = datetime.strptime(f_test_name,'%Y%m%d')
        
        # REMOVE THE ICME EVENTS
        if solar_date_test not in full_icme_list:
            # THEN APPEND THE DATES INTO A LIST
            map_test_non_icme_date.append(solar_date_test)
            
        
    return map_train_non_icme_date, map_test_non_icme_date

def time_adaptation_delay_prep(train_start_date, train_end_date, test_start_date, test_end_date, train_speed_dates, test_speed_dates, map_train_non_icme_date, map_test_non_icme_date):
    """
    Prepares the SW (train and test) and Maps (train and test) Index.

    Parameters:
    ----------
    train_start_date : str
        The start time of the training period, formatted as a string (e.g., "YYYY-MM-DD").
    train_end_date : str
        The end time of the training period, formatted as a string (e.g., "YYYY-MM-DD").
    test_start_date : str
        The start time of the testing period, formatted as a string (e.g., "YYYY-MM-DD").
    test_end_date : str
        The end time of the testing period, formatted as a string (e.g., "YYYY-MM-DD").    
    train_speed_dates : list,
        The SW speed dates within the study train period.
    test_speed_dates : list,
        The SW speed dates within the study test period.
    map_train_non_icme_date : list,
        The Filtered Train Map date within the study train period.
    map_test_non_icme_date : list,
        The Filtered Test Map date within the study test period.
    Returns:
    -------
    Speed_train_indices : list,
    Speed_test_indices : list,
    solar_image_train_indices : list,
    solar_image_test_indices : list.
    
    Examples:
    ---------
    >>> time_adaptation_delay_prep(train_start_date="2010-05-13",
                                    train_end_date="2011-07-06",
                                    test_start_date="2011-07-07",
                                    test_end_date="2011-12-31", 
                                    train_speed_dates, 
                                    test_speed_dates, 
                                    map_train_non_icme_date, 
                                    map_test_non_icme_date
        )
    """
    
    Train_date_range = pd.date_range(start=f'{train_start_date}', end=f'{train_end_date}')
    Test_date_range = pd.date_range(start= f'{test_start_date}', end= f'{test_end_date}')

    # CREATE THE DATE RANGE INDEX
    Train_date_to_index = {date: idx for idx, date in enumerate(Train_date_range)}
    Test_date_to_index = {date: idx for idx, date in enumerate(Test_date_range)}

    # CREATE THE DATE RANGE INDEX - FOR THE NON ICME SOLAR WIND VELOCITy
    Speed_train_indices = [Train_date_to_index[date] for date in train_speed_dates]
    Speed_test_indices = [Test_date_to_index[date] for date in test_speed_dates]

    # CREATE THE DATE RANGE INDEX - FOR THE NON ICME SOLAR CORONA BINARY IMAGES
    solar_image_train_indices = [Train_date_to_index[date] for date in pd.to_datetime(map_train_non_icme_date)]
    solar_image_test_indices = [Test_date_to_index[date] for date in pd.to_datetime(map_test_non_icme_date)]
    
    return Speed_train_indices, Speed_test_indices, solar_image_train_indices, solar_image_test_indices

def match_data_prep(num_delay, timesteps, map_images_dates, SW_dates, Map_indices, SW_indices):
    """
    returns the map and SW date list based on Time Delays matching..

    Parameters:
    ----------
    num_delay : int,
        Time delay ranging from (0 to 4) ---> where indexing starts from 0.
    timesteps : int, 
        Timestep is 1. 
    map_images_dates : list,
        The Filtered Map date within the study Train/Test period. 
    SW_dates : list,
        The SW speed dates within the study train/test period.
    Map_indices : list,
        The Filtered Map index within the study Train/Test period.
    SW_indices : list,
        The SW speed index within the study train/test period.
        
    Returns:
    -------
    input_image_date_list : list,
    output_SW_dates_list : list.
    
    Examples:
    ---------
    >>> match_data_prep(num_delay, 
                      timesteps=1, 
                      map_images_dates, 
                      SW_dates, 
                      Map_indices, 
                      SW_indices
        )
    """
    
    input_image_date_list = []
    input_SW_dates_list = []
    output_SW_dates_list = []

    #loop through the image indices
    for i in range(len(Map_indices)):
        a = Map_indices[i]
        
        # Loop through the SW indice w.r.t the time delay 
        for j in range(len(SW_indices) - (num_delay + timesteps)):
            # If the image index matches with the (SW speed index + delay)
            if SW_indices[j] == a and SW_indices[j + (num_delay + timesteps)] == a + (num_delay + timesteps):
                
                image__date = map_images_dates[i]
                input_image_date_list.append(image__date)

                # input_SW_dates = SW_dates[j:j + timesteps]
                # input_SW_dates_list.append(input_SW_dates)

                output_SW_dates = SW_dates[j + (num_delay + timesteps)]
                output_SW_dates_list.append(output_SW_dates)

    return input_image_date_list, output_SW_dates_list

def filter_output(df, Train_output_timeseries_date, Test_output_timeseries_date):
    """
    Returns the Train, Test SW speeds and the Test date, based on the Time delay match between the SW and Map.

    Parameters:
    ----------
    df : pandas.DataFrame,
        SW speed dataframe.
    Train_output_timeseries_date : list,
        List of the SW speed training dataset date (after time delay matching).
    Test_output_timeseries_date : list,
        List of the SW speed testing dataset date (after matching delay matching).
        
    Returns:
    -------
    train_velocity_out : numpy.ndarray
        The SW speed train as a NumPy array.
    test_velocity_out : numpy.ndarray
        The SW speed test as a NumPy array.
    test_timestamps_output : pandas.DatetimeIndex
        The SW speed test time strings converted to pandas datetime objects.
    
    Examples:
    ---------
    >>> filter_output(df,
                      Train_output_timeseries_date,
                      Test_output_timeseries_date
        )
    """
    
    df["DATE"] = pd.to_datetime(df["DATE"], format="mixed") 
    train_timestamps_output = pd.to_datetime(Train_output_timeseries_date)
    train_velocity_output = df[df["DATE"].isin(train_timestamps_output)]
    train_velocity_out = np.array(train_velocity_output[["V(Km/s)"]])
  
    test_timestamps_output = pd.to_datetime(Test_output_timeseries_date)
    test_velocity_output = df[df["DATE"].isin(test_timestamps_output)]
    test_velocity_out = np.array(test_velocity_output[["V(Km/s)"]])

    return train_velocity_out, test_velocity_out, test_timestamps_output

def train_test_set(img_train_path, img_test_path, Train_input_image_date_list, Test_input_image_date_list):
    
    """
    Returns the Train, and Test Maps, based on the Time delay match between the SW and Map.

    Parameters:
    ----------
    img_train_path : str
        The root directory or file path where the Map training set are located.
    img_test_path : 
        The root directory or file path where the Map testing set are located.
    Train_input_image_date_list : list,
        New Image dates based on the time delay match (Train set). 
    Test_input_image_date_list : list,
        New Image dates based on the time delay match (Test set). 
      
    Returns:
    -------
    train_images : numpy.ndarray
        NumPy array of the Map (Train Set).
    test_images : numpy.ndarray
        NumPy array of the Map (Test Set).
    Examples:
    ---------
    >>> train_test_set(img_train_path, 
                       img_test_path,
                       Train_input_image_date_list, 
                       Test_input_image_date_list
        )
    """
    train_filename = sorted(glob.glob(f"{img_train_path}/**"))
    train_images = []
  
    for train_file in train_filename:
        #  Extract the Date of the Image
        f_name = train_file[-8:]
    
        # Convert the String Date to Datetime Format
        solar_date_train = datetime.strptime(f_name,'%Y%m%d')
    
        # SELECT ONLY THE TRAIN IMAGE BASED ON THE DATE
        if solar_date_train in Train_input_image_date_list:
            # Load the fit file
            train_img = fits.open(train_file)
            # Extract the Image data from the hdu fit file
            train_aia_img = train_img[0].data

            # Append 
            train_images.append(train_aia_img)
    
    test_filename = sorted(glob.glob(f"{img_test_path}/**"))
    test_images = []
    
    for test_file in test_filename:
        
        #  Extract the Date of the Image 
        f_test_name = test_file[-8:]
        
        # Convert the String Date to Datetime Format
        solar_date_test = datetime.strptime(f_test_name,'%Y%m%d')
        
        # SELECT ONLY THE TEST IMAGE BASED ON THE DATE
        if solar_date_test in Test_input_image_date_list:
            
            # Load the fit file
            test_img = fits.open(test_file)
            # Extract the Image data from the hdu fit file
            test_aia_img = test_img[0].data 

            # Append 
            test_images.append(test_aia_img)
      
    return np.array(train_images), np.array(test_images)


def reduced_chi_squared(y_true, y_pred, sigma):
    """
    Calculate the reduced chi-squared (reduced mean square error)
    
    Parameters:
    ----------
    y_true : array-like : True observed values
    y_pred : array-like : Predicted values
    sigma  : array-like : Standard deviation (errors) of the true values
    
    Returns:
    ----------
    float : Reduced chi-squared value

    Examples:
    ---------
    >>> reduced_chi_squared(y_true, 
                           y_pred,
                           sigma, 
        )
    """
    
    if len(y_true) != len(y_pred) or len(y_true) != len(sigma):
        raise ValueError("Input arrays must have the same length")

    # Calculate the reduced chi-squared
    chi_squared = np.sum(((y_true - y_pred) ** 2) / (sigma ** 2))
    reduced_chi_squared = chi_squared / len(y_true)
    
    return reduced_chi_squared

def SpeedNet_BM():
    """
    SpeedNet-BM model. 
    Contains CNN-SpeedNet (LSTM-FFNN) architecture for spatial convolution of a binary map with shape 256, 256, 1

    
    Parameters:
    ----------
    None.
    
    Returns:
    ----------
    combined_model : Combine model architecture
    
    Examples:
    ---------
    >>> combined_model = SpeedNet_BM()
    """
    tf.keras.backend.clear_session()
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    cnn_input = Input(shape=(256, 256, 1))
    x = Conv2D(16, (3, 3), activation='relu')(cnn_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    cnn_output = Flatten()(x)

    # Define the LSTM and FFNN Layer 
    cnn_output_reshaped = Reshape((1, cnn_output.shape[1]))(cnn_output)
    lstm_output = LSTM(30, activation='relu')(cnn_output_reshaped)
    dense_lay = Dense(200, activation='relu')(lstm_output)
    lstm_output = Dense(1, activation='linear')(dense_lay)
    
    # Combine CNN and LSTM into a single model
    combined_model = Model(inputs=cnn_input, outputs=lstm_output)
    return combined_model


def SpeedNet_EUV():
    """
    SpeedNet-EUV model. 
    Contains CNN-SpeedNet (LSTM-FFNN) architecture for spatial convolution of a EUV map with shape 256, 256, 3

    
    Parameters:
    ----------
    None.
    
    Returns:
    ----------
    combined_model : Combine model architecture
    
    Examples:
    ---------
    >>> combined_model = SpeedNet_EUV()
    """
    tf.keras.backend.clear_session()
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    cnn_input = Input(shape=(256, 256, 3))
    x = Conv2D(16, (3, 3), activation='relu')(cnn_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    cnn_output = Flatten()(x)
    
    # Define the LSTM and FFNN Layer 
    cnn_output_reshaped = Reshape((1, cnn_output.shape[1]))(cnn_output)
    lstm_output = LSTM(30, activation='relu')(cnn_output_reshaped)
    dense_lay = Dense(200, activation='relu')(lstm_output)
    lstm_output = Dense(1, activation='linear')(dense_lay)
    
    # Combine CNN and LSTM into a single model
    combined_model = Model(inputs=cnn_input, outputs=lstm_output)
    return combined_model


def SpeedNet_WAve():
    """
    SpeedNet-WAve model. 
    Contains CNN-SpeedNet (LSTM-FFNN) architecture for spatial convolution of a EUV map with shape 256, 256, 3
    This model considers that each EUV(171, 193, 304) should be treated as a function of it' features.
    Therefore It must should not be normalized with other Wavelengths
    
    Parameters:
    ----------
    None.
    
    Returns:
    ----------
    combined_model : Combine model architecture
    
    Examples:
    ---------
    >>> combined_model = SpeedNet_WAve()
    """
    tf.keras.backend.clear_session()
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    cnn_input = Input(shape=(256, 256, 3))
    x = Conv2D(16, (3, 3), activation='relu')(cnn_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    cnn_output = Flatten()(x)
    
    # Define the LSTM and FFNN Layer 
    cnn_output_reshaped = Reshape((1, cnn_output.shape[1]))(cnn_output)
    lstm_output = LSTM(30, activation='relu')(cnn_output_reshaped)
    dense_lay = Dense(200, activation='relu')(lstm_output)
    lstm_output = Dense(1, activation='linear')(dense_lay)
    
    # Combine CNN and LSTM into a single model
    combined_model = Model(inputs=cnn_input, outputs=lstm_output)
    return combined_model
