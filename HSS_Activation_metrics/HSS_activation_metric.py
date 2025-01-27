import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


def gradcamm(img_array, best_model):
    # predss = best_model.predict(img_array)
    
    # First part
    last_conv_layer_name = "max_pooling2d_3"
    last_conv_layer = best_model.get_layer(last_conv_layer_name)
    last_conv_layer_model = Model(best_model.inputs, last_conv_layer.output)
    
    # Second part _ LSTM Regressor
    regression_layer_names = [
       "flatten",
       "reshape",
       "lstm",
       "dense",
       "dense_1"
    ]
    
    lstm_regressor_input = Input(shape=(14, 14, 128))
    x = lstm_regressor_input
    for layer_name in regression_layer_names:
        x = best_model.get_layer(layer_name)(x)
    regressor_model = Model(lstm_regressor_input, x)
    
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img_array)
        preds = regressor_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
        
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2)).numpy()
    
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    for i in range(128):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
        
    heatmap = np.sum(last_conv_layer_output, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    resized_heatmap = cv2.resize(np.array(heatmap), (256,256))
    return resized_heatmap, preds
    

def HSS_activation_csv(channel,full_period_test_velocity_out, full_period_test_images, full_period_Test_timeseries_date, full_period_num_delay, best_model_full_period):
    
    """
    Prepares the Mean Activation, Mean Activation and Maximum Activation across specified Time Delay.
    
    """
    SN_list = []
    Mean_list = []
    max_list = []
    min_list = []
    Original_solar_wind = []
    predicted_solar_wind = []
    wind_date = []
    
    full_period_max = np.argmax(full_period_test_velocity_out)

    longer_HSS_full_period_indices_rand = [full_period_max-4, full_period_max-3, full_period_max-2, full_period_max-1, full_period_max, full_period_max+1, full_period_max+2, full_period_max+3, full_period_max+4]
   
    Longer_HSS_full_period_random_selection = full_period_test_images[longer_HSS_full_period_indices_rand]
    
    
    if channel == 1:
        for i in range(len(Longer_HSS_full_period_random_selection)):
            Hss_full_period_img_cam, Hss_full_period_value = gradcamm(Longer_HSS_full_period_random_selection[i].reshape(1,256, 256, 1), best_model_full_period)
            SN_list.append(i+1)
            wind_date.append(list(full_period_Test_timeseries_date)[longer_HSS_full_period_indices_rand[i]+(full_period_num_delay+1)])
            Mean_list.append(np.mean(Hss_full_period_img_cam))
            max_list.append(np.max(Hss_full_period_img_cam))
            min_list.append(np.min(Hss_full_period_img_cam))
            Original_solar_wind.append(full_period_test_velocity_out[longer_HSS_full_period_indices_rand[i]])
            predicted_solar_wind.append(Hss_full_period_value) 
        
    else:
        for i in range(len(Longer_HSS_full_period_random_selection)):
            Hss_full_period_img_cam, Hss_full_period_value = gradcamm(Longer_HSS_full_period_random_selection[i].reshape(1,256, 256, 3), best_model_full_period)
            SN_list.append(i+1)
            wind_date.append(list(full_period_Test_timeseries_date)[longer_HSS_full_period_indices_rand[i]+(full_period_num_delay+1)])
            Mean_list.append(np.mean(Hss_full_period_img_cam))
            max_list.append(np.max(Hss_full_period_img_cam))
            min_list.append(np.min(Hss_full_period_img_cam))
            Original_solar_wind.append(full_period_test_velocity_out[longer_HSS_full_period_indices_rand[i]])
            predicted_solar_wind.append(Hss_full_period_value)
        
    activation_df = pd.DataFrame({
            'Map_index': SN_list,
            'Wind_date': wind_date,
            'Mean_act' : Mean_list,
            'Max_act': max_list,
            'Min_act': min_list,
            'Original_HSS': tf.squeeze(Original_solar_wind),
            'Predicted_HSS': tf.squeeze(predicted_solar_wind)
        })
    if channel == 1:
        activation_df.to_csv(f'Binary_HSS_Gradcam_activation_metrics_delay_{full_period_num_delay}.csv')
    else:
        activation_df.to_csv(f'Channel_HSS_Gradcam_activation_metrics_delay_{full_period_num_delay}.csv')
  