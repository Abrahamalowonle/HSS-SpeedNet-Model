import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Get the current script's directory
current_folder = os.path.dirname(os.path.abspath(__file__))

hsemetric_folder = os.path.abspath(os.path.join(current_folder, "../HSEMETRIC"))
sys.path.append(hsemetric_folder)


# Now import the module
import hsemetric        #(This package was created by Upendra et al. 2020)
import SpeedNet_model_prep

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from keras.models import load_model


def SpeedNet_model_training(path, img_train_path, img_test_path, train_start_date, train_end_date, test_start_date, test_end_date, model_path):
    """
    SpeedNet_model training function.
    This function calls the necessary function for the SpeedNet training of the maps (either SpeedNet-EUV or SpeedNet-BM).
    Across the different Time delays and Cross-Validation Folds.
    
    Parameters:
    ----------
    path : str
        The root directory or file path where the Filtered SW, standard deviation of solar wind, ICME_event files are located.
    img_train_path : str
        The root directory or file path where the Map training set are located.
    img_test_path : 
        The root directory or file path where the Map testing set are located.
    train_start_date : str
        The start time of the training period, formatted as a string (e.g., "YYYY-MM-DD").
    train_end_date : str
        The end time of the training period, formatted as a string (e.g., "YYYY-MM-DD").
    test_start_date : str
        The start time of the testing period, formatted as a string (e.g., "YYYY-MM-DD").
    test_end_date : str
        The end time of the testing period, formatted as a string (e.g., "YYYY-MM-DD").
    model_path : str
        The root directory where the trained model,and plots would be saved.
    Returns:
    -------
    None.
    
    Examples:
    ---------
    >>> combined_model = SpeedNet_model_training(path="/data/files",
                                                 train_start_date="2010-05-13",
                                                 train_end_date="2011-07-06",
                                                 test_start_date="2011-07-07",
                                                 test_end_date="2011-12-31",
                                                 model_path="data/maximum")
    
    """
    
    n_steps = 1
    # The Timestep is 1. while the Time delay ranges from 0 to 4 index. (i.e delay of 1 to 5)
    for num_delay in range(5):
        df, velocity_train, velocity_test, std_velocity_test, train_speed_dates, test_speed_dates, full_icme_list = SpeedNet_model_prep.load_df(
                                                                                                                    path, train_start_date,
                                                                                                                    train_end_date, test_start_date, 
                                                                                                                    test_end_date)
        
        map_train_non_icme_date, map_test_non_icme_date = SpeedNet_model_prep.non_icme_Image_date(img_train_path, 
                                                                                              img_test_path,
                                                                                              full_icme_list)
        
        Speed_train_indices, Speed_test_indices, solar_image_train_indices, solar_image_test_indices = SpeedNet_model_prep.time_adaptation_delay_prep(
                                                                                                       train_start_date, train_end_date,
                                                                                                       test_start_date, test_end_date, 
                                                                                                       train_speed_dates, test_speed_dates,
                                                                                                       map_train_non_icme_date, map_test_non_icme_date)
        
        New_Train_Map_date, New_Train_SW_date = SpeedNet_model_prep.match_data_prep(num_delay, n_steps, map_train_non_icme_date, train_speed_dates,
                                                                               solar_image_train_indices, Speed_train_indices)    
        
        New_Test_Map_date, New_Test_SW_date = SpeedNet_model_prep.match_data_prep(num_delay, n_steps, map_test_non_icme_date, test_speed_dates,
                                                                             solar_image_test_indices, Speed_test_indices)
    
    
        train_velocity_out, test_velocity_out, test_timestamps_output = SpeedNet_model_prep.filter_output(df, New_Train_SW_date, New_Test_SW_date)
    
        train_images, test_images =  SpeedNet_model_prep.train_test_set(img_train_path, img_test_path,
                                                                New_Train_Map_date, New_Test_Map_date)
        n_splits = 5
        for fold_index in range(5):
            num_samples = len(train_images)
            val_start = fold_index * (num_samples // n_splits)
            val_end = val_start + (num_samples // n_splits)
            
            if val_end > num_samples - 1:
                val_end = num_samples - 1
                
            # Define the training and validation sets
            X_images_train, X_images_val= np.concatenate([train_images[:val_start], train_images[val_end:num_samples-1]]), train_images[val_start:val_end]
            
            y_speed_train, y_speed_val = np.concatenate([train_velocity_out[:val_start], train_velocity_out[val_end:num_samples-1]]), train_velocity_out[val_start:val_end]
           
            if len(X_images_train.shape) < 4:
                return f"Incompatible shape."
    
            # If Shape[3] == 1, Utilized SpeedNet-BM model
            if X_images_train.shape[3] == 1:
                combined_model =  SpeedNet_model_prep.SpeedNet_BM()
                
            # If Shape[3] == 3, Utilized SpeedNet-EUV model
            elif X_images_train.shape[3] == 3:
                combined_model =  SpeedNet_model_prep.SpeedNet_EUV()
                
            else:
                return "Shape doesn't match the expected values."

            # Compile the combined model
            initial_learning_rate = 0.0001
            optimizer = Adam(learning_rate=initial_learning_rate)
            combined_model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
            
            tf.keras.utils.plot_model(combined_model, to_file= f"{model_path}/model_Diagram_fold_{fold_index + 1}_nstep_{n_steps}_delay_{num_delay}.png", show_shapes=True)
            
            checkpoint_callback = ModelCheckpoint(filepath=f"{model_path}/best_model_for_fold_{fold_index + 1}_timestep_{n_steps}_and_delay_{num_delay}.keras",
                                                  monitor='val_loss', save_best_only=True, 
                                                  mode='min',
                                                  verbose=1)
            
            # Train the model with validation
            
            history = combined_model.fit(X_images_train, y_speed_train,
                                         epochs=100, batch_size=12, 
                                         validation_data=(X_images_val, y_speed_val),
                                         callbacks=[checkpoint_callback]) 
            
            # plot the training and validation accuracy and loss at each epoch
            fig, ax= plt.subplots(nrows=1, ncols=1, figsize=(20,5), sharex=True)
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            epochs = range(1, len(loss) + 1)
            ax.plot(epochs, loss, 'y', label='Training loss')
            ax.plot(epochs, val_loss, 'r', label='Validation loss')
            plt.title(f'Training and validation loss for_fold_{fold_index + 1}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            ax.legend()
            plt.savefig(f"{model_path}/Loss_curve_for_fold_{fold_index + 1}_timestep_{n_steps}_and_delay_{num_delay}.png");
            plt.close()

def SpeedNet_model_testing(path, img_train_path, img_test_path, train_start_date, train_end_date, test_start_date, test_end_date, model_path):
    """
    SpeedNet_model testing function.
    This function the SpeedNet model (either SpeedNet-EUV or SpeedNet-BM) across the testing set and saves the metrics table.
    Across the different Time delays and Cross-Validation Folds.
    
    Parameters:
    ----------
    path : str
        The root directory or file path where the Filtered SW, standard deviation of solar wind, ICME_event files are located.
    img_train_path : str
        The root directory or file path where the Map training set are located.
    img_test_path : 
        The root directory or file path where the Map testing set are located.
    train_start_date : str
        The start time of the training period, formatted as a string (e.g., "YYYY-MM-DD").
    train_end_date : str
        The end time of the training period, formatted as a string (e.g., "YYYY-MM-DD").
    test_start_date : str
        The start time of the testing period, formatted as a string (e.g., "YYYY-MM-DD").
    test_end_date : str
        The end time of the testing period, formatted as a string (e.g., "YYYY-MM-DD").
    model_path : str
        The root directory where the trained model,and plots would be saved.
    Returns:
    -------
    None.
    
    Examples:
    ---------
    >>> combined_model = SpeedNet_model_training(path="/data/files",
                                                 train_start_date="2010-05-13",
                                                 train_end_date="2011-07-06",
                                                 test_start_date="2011-07-07",
                                                 test_end_date="2011-12-31",
                                                 model_path="data/maximum")
    
    """
    full_metric_list = []
    n_steps = 1
    n_splits = 5
    # The Timestep is 1. while the Time delay ranges from 0 to 4 index. (i.e delay of 1 to 5)
    for num_delay in range(5):
        for fold_index in range(n_splits):
            df, velocity_train, velocity_test, std_velocity_test, train_speed_dates, test_speed_dates, full_icme_list = SpeedNet_model_prep.load_df(
                                                                                                                        path, train_start_date,
                                                                                                                        train_end_date, test_start_date, 
                                                                                                                        test_end_date)
            
            map_train_non_icme_date, map_test_non_icme_date = SpeedNet_model_prep.non_icme_Image_date(img_train_path, 
                                                                                                img_test_path,
                                                                                                full_icme_list)
            
            Speed_train_indices, Speed_test_indices, solar_image_train_indices, solar_image_test_indices = SpeedNet_model_prep.time_adaptation_delay_prep(
                                                                                                        train_start_date, train_end_date,
                                                                                                        test_start_date, test_end_date, 
                                                                                                        train_speed_dates, test_speed_dates,
                                                                                                        map_train_non_icme_date, map_test_non_icme_date)
            
            New_Train_Map_date, New_Train_SW_date = SpeedNet_model_prep.match_data_prep(num_delay, n_steps, map_train_non_icme_date, train_speed_dates,
                                                                                solar_image_train_indices, Speed_train_indices)    
            
            New_Test_Map_date, New_Test_SW_date = SpeedNet_model_prep.match_data_prep(num_delay, n_steps, map_test_non_icme_date, test_speed_dates,
                                                                                solar_image_test_indices, Speed_test_indices)
        
        
            train_velocity_out, test_velocity_out, test_timestamps_output = SpeedNet_model_prep.filter_output(df, New_Train_SW_date, New_Test_SW_date)

            
            _, std_vel_test, _ = SpeedNet_model_prep.filter_output(std_velocity_test, New_Train_SW_date, New_Test_SW_date)
            
            train_images, test_images =  SpeedNet_model_prep.train_test_set(img_train_path, img_test_path,
                                                                    New_Train_Map_date, New_Test_Map_date)

            loaded_model = load_model(f"{model_path}/best_model_for_fold_{fold_index + 1}_timestep_{n_steps}_and_delay_{num_delay}.keras")

            predictions = loaded_model.predict(test_images)
            
            best_mae = mean_absolute_error(tf.squeeze(test_velocity_out), tf.squeeze(predictions))
            best_mse = mean_squared_error(tf.squeeze(test_velocity_out), tf.squeeze(predictions))
            best_rmse = root_mean_squared_error(tf.squeeze(test_velocity_out), tf.squeeze(predictions))
            corr_matrix = np.corrcoef(tf.squeeze(test_velocity_out), tf.squeeze(predictions))
            corr_coef = corr_matrix[0, 1]
            
            # Fisher transformation for uncertainty
            z = 0.5 * np.log((1 + corr_coef) / (1 - corr_coef))
            n = len(tf.squeeze(test_velocity_out))
            se = 1 / np.sqrt(n - 3)
            z_conf_interval = se * 1.96  # 95% confidence interval
            pearson_conf_interval = 1.96 * (1 / np.sqrt(n - 3))
            pearson_uncertainty = np.tanh(z_conf_interval)
            # Uncertainty in MSE
            mse_uncertainty = np.sqrt(2 * best_mse ** 2 / n)

            # Uncertainty in RMSE
            rmse_uncertainty = 0.5 * mse_uncertainty / best_rmse
            
            tp,fp,fn= hsemetric.BatchwiseHSE(test_velocity_out[(n_steps-1):], np.array(predictions))
            threat_score=tp*1.0/(tp+fp+fn)
            
            reduced_chi_squ = SpeedNet_model_prep.reduced_chi_squared(test_velocity_out[(n_steps-1):], np.array(predictions), std_vel_test[(n_steps-1):])
            
            
            data_pred = {
                    'date': test_timestamps_output[(n_steps-1):],
                    'test_original': tf.squeeze(test_velocity_out[(n_steps-1):]),
                    'test_predicted': tf.squeeze(predictions)
                }

            # Create DataFrame
            pred_df = pd.DataFrame(data_pred)

            # Create a complete date range from the minimum to the maximum date in the data
            pred_date_range = pd.date_range(start=pred_df['date'].min(), end=pred_df['date'].max())

            # Reindex the DataFrame to insert NaN for missing dates
            pred_df = pred_df.set_index('date').reindex(pred_date_range).rename_axis('date').reset_index()

            # %matplotlib inline
            # Plot the data
            fig, ax= plt.subplots(nrows=1, ncols=1, figsize=(20,5), sharex=True)
            plt.plot()

            fig.suptitle(f"Solar Wind Prediction Fold_{fold_index + 1}_timestep_{n_steps}_and_delay_{num_delay}", fontsize = 16, fontweight = "bold");

            ax.plot(pred_df['date'],pred_df['test_predicted'], linestyle='-', color='g', label= f'Prediction MAE:{best_mae} \n RMSE: {best_rmse} \n Corr: {corr_coef:.3f} ± {pearson_uncertainty:.3f}')

            ax.grid(False)

            ax.legend()
            
            plt.savefig(f"{model_path}/predictions_for_fold_{fold_index + 1}_and_step_{n_steps}_and_delay_{num_delay}.png");
            plt.close()
            

            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=1, cbar=True)
            plt.title(f'Correlation Matrix for Fold {fold_index + 1} and Step {n_steps} and Delay {num_delay}')

            plt.savefig(f"{model_path}/Pearson_corr_for_fold_{fold_index + 1}_and_step_{n_steps}_and_delay_{num_delay}.png");
            plt.close()
            
            row = ((fold_index + 1), n_steps, (num_delay+1), best_mae, f"{best_mse:.3f} ± {mse_uncertainty:.3f}", f"{best_rmse:.3f} ± {rmse_uncertainty:.3f}", f"{corr_coef:.3f} ± {pearson_uncertainty:.3f}", f"{threat_score}", f"{reduced_chi_squ}") 
            full_metric_list.append(row)

    full_mae_list_df = pd.DataFrame(full_metric_list, columns = ['Cross_Folds','Timestep','Delay','Mae', "MSE", 'RMSE', "Corr Coef", "Threat_score", "Reduced Chi Square"])   
    full_mae_list_df.to_excel(f"{model_path}/full_metrics.xlsx")

    