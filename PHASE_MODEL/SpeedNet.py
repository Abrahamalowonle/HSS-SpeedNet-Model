# ===============================
# PACKAGES
# ===============================
import os, re
import sys
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from datetime import timedelta
from astropy.io import fits
from scipy.stats import pearsonr

import gc
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Reshape, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE
from sklearn.model_selection import KFold

# # ===============================
# # Try import hsemetric
# # ===============================
# try:
#     current_folder = os.getcwd()
#     hsemetric_folder = os.path.abspath(os.path.join(current_folder, "../HSEMETRIC"))
#     sys.path.append(hsemetric_folder)
#     import hsemetric
# except Exception:
#     hsemetric = None
#     print("hsemetric not available — Threat_score will be set to NaN.")


# ===============================
# Reduced Chi-Square
# ===============================
def reduced_chi_squared(y_true, y_pred, sigma):
    """
    Calculate the reduced chi-squared (goodness-of-fit statistic).

    Parameters
    ----------
    y_true : array-like
        Observed values (ground truth).
    y_pred : array-like
        Predicted values from the model.
    sigma : array-like
        Standard deviation (errors) corresponding to observed values.

    Returns
    -------
    float
        Reduced chi-squared value.
    """
    if len(y_true) != len(y_pred) or len(y_true) != len(sigma):
        raise ValueError("Input arrays must have the same length")
    chi_squared = np.sum(((y_true - y_pred) ** 2) / (sigma ** 2))
    return chi_squared / len(y_true)


# =========================
# SpeedNet Model Definition
# =========================
def SpeedNet(input_shape, lstm_unit=10, dense_unit=100):
    """
    Define SpeedNet CNN+LSTM model for solar wind speed prediction.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input image (H, W, C).
    lstm_unit : int
        Number of LSTM units.
    dense_unit : int
        Number of dense units before the final regression layer.

    Returns
    -------
    tf.keras.Model
        Compiled CNN-LSTM model.
    """
    tf.keras.backend.clear_session()
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    cnn_input = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), activation="relu")(cnn_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    cnn_output = Flatten()(x)

    cnn_output_reshaped = Reshape((1, cnn_output.shape[1]))(cnn_output)
    lstm_output = LSTM(lstm_unit, activation="relu")(cnn_output_reshaped)
    dense_lay = Dense(dense_unit, activation="relu")(lstm_output)
    lstm_output = Dense(1, activation="linear")(dense_lay)

    return Model(inputs=cnn_input, outputs=lstm_output)


# =========================
# Data Preparation
# =========================
def prepare_dataset(sw_csv, icme_excel, img_path, std_csv):
    """
    Load and merge solar wind speed, standard deviation, and ICME list.

    Parameters
    ----------
    sw_csv : str
        Path to solar wind CSV file.
    icme_excel : str
        Path to ICME Excel file.
    img_path : str
        Directory containing EUV image files.
    std_csv : str
        Path to CSV containing standard deviation values.

    Returns
    -------
    tuple
        (merged dataframe, dict of date->image_file, icme_dates set)
    """
    df_sw = pd.read_csv(sw_csv)
    df_sw["DATE"] = pd.to_datetime(df_sw["DATE"])
    df_sw = df_sw.rename(columns={"V(Km/s)": "speed"})

    df_std = pd.read_csv(std_csv)
    df_std["DATE"] = pd.to_datetime(df_std["DATE"])
    df_std = df_std.rename(columns={"V(Km/s)": "std_speed"})

    df = pd.merge(df_sw, df_std, on="DATE", how="inner")

    icme_list = pd.read_excel(icme_excel)
    icme_list["Start_time"] = pd.to_datetime(icme_list["Start_time"].str.split(" ").str[0], format="%Y/%m/%d")
    icme_list["End_time"] = pd.to_datetime(icme_list["End_time"].str.split(" ").str[0], format="%Y/%m/%d")

    icme_dates = []
    for start, end in zip(icme_list["Start_time"], icme_list["End_time"]):
        icme_dates.extend(pd.date_range(start, end).tolist())
    icme_dates = set(icme_dates)

    df = df[~df["DATE"].isin(icme_dates)].reset_index(drop=True)

    img_files, img_dates = [], []
    for f in os.listdir(img_path):
        if f.isdigit() and len(f) == 8:
            date = pd.to_datetime(f, format="%Y%m%d")
        else:
            m = re.search(r"(\d{8})", f)
            if not m:
                continue
            date = pd.to_datetime(m.group(1), format="%Y%m%d")

        if date in icme_dates:
            continue
        img_files.append(f)
        img_dates.append(date)

    img_dates, img_files = zip(*sorted(zip(img_dates, img_files)))
    return df, dict(zip(img_dates, img_files)), icme_dates


# =========================
# Match SW + STD to EUV image
# BACKTRACKING MAPPING (Image_time = SW time - delay)
# img_date = sw_date - timedelta(days=delay)
# =========================
def match_data(df, img_dict, delay):
    """
    Match solar wind + std to EUV images with forecast delay.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing DATE, speed, std_speed.
    img_dict : dict
        Mapping from date -> image filename.
    delay : int
        Forecast delay (days).

    Returns
    -------
    pd.DataFrame
        Matched dataset with sw_date, img_date, file, speed, std_speed.
    """
    matched = []
    for _, row in df.iterrows():
        sw_date, sw_speed, std_speed = row["DATE"], row["speed"], row["std_speed"]
        img_date = sw_date - timedelta(days=delay)
        if img_date in img_dict:
            matched.append((sw_date, img_date, img_dict[img_date], sw_speed, std_speed))
    return pd.DataFrame(matched, columns=["sw_date", "img_date", "img_file", "speed", "std_speed"])


# =========================
# Load FITS Map
# Either Binary Map or Multi-wavelength
# =========================
def load_map(filepath):
    """
    Load a FITS map and return as numpy array.

    Parameters
    ----------
    filepath : str
        Path to FITS file.

    Returns
    -------
    np.ndarray
        Image array with shape (H, W, C).
    """
    with fits.open(filepath) as fit_img:
        arr = fit_img[0].data
    arr_np = np.array(arr, dtype=np.float32)
    if arr_np.ndim == 2:
        arr_np = np.expand_dims(arr_np, -1)
    return arr_np


# ===============================
# Plotting Function
# ===============================

def plot_results(y_test, preds, test_batches, icme_dates, delay, fold_index,
                 threat_score, corr, p_value, save_path,  model_type):
    """
    Plot predicted vs observed solar wind speed and correlation plot.

    Parameters
    ----------
    y_test : array-like
        Ground truth solar wind speed values.
    preds : array-like
        Predicted solar wind speed values.
    test_batches : list of DataFrames
        Batches of test data containing 'sw_date'.
    icme_dates : list of datetime-like
        Dates of ICME events for reference in the time series plot.
    delay : int
        Forecast delay (in days).
    fold_index : int
        Cross-validation fold index.
    threat_score : float
        Threat score calculated from hsemetric (if available).
    corr : float
        Pearson correlation coefficient between observed and predicted values.
    p_value : float
        p-value corresponding to the Pearson correlation.
    save_path : str
        File path to save the generated figure.
    """
    df_plot = pd.DataFrame({
        "sw_date": pd.to_datetime(test_batches["sw_date"].values),
        "observed": y_test,
        "predicted": preds.flatten()
    }).sort_values("sw_date")
    

    # Create a complete date range from the minimum to the maximum date in the data
    pred_date_range = pd.date_range(start=df_plot['sw_date'].min(), end=df_plot['sw_date'].max())

    # Reindex the DataFrame to insert NaN for missing dates
    df_plot = df_plot.set_index('sw_date').reindex(pred_date_range).rename_axis('sw_date').reset_index()


    fig, ax = plt.subplots(1, 2, figsize=(20, 5), gridspec_kw={'width_ratios': [3, 1]})
    plt.style.use("default")
    fig.suptitle(f"SpeedNet-{model_type} SW Prediction VS Observed (Fold {fold_index}, Delay {delay})",
                 fontsize=16, fontweight="bold", y=1.0)

    # --- Time series plot
    ax[0].plot(df_plot["sw_date"], df_plot["predicted"], linestyle='-', color='b', linewidth=2, label="PREDICTED")
    ax[0].plot(df_plot["sw_date"], df_plot["observed"], linestyle='-', color='g', linewidth=2, label="OBSERVED")

    for i, icme_date in enumerate(icme_dates):
        if df_plot["sw_date"].min() <= icme_date <= df_plot["sw_date"].max():
            ax[0].axvline(x=icme_date, color="black", linestyle="-", ymax=1, ymin=0.95, linewidth=2.5,
                          label="ICME Event" if i == 0 else "")

    ax[0].set_title(f"Testset: Delay {delay}", fontsize=15, fontweight="bold")
    ax[0].set_xlabel("SW DATE", fontsize=15, fontweight="bold")
    ax[0].set_ylabel("V (Km/s)", fontsize=15, fontweight="bold")
    ax[0].axhline(y=300, color="red", linestyle="--", linewidth=2, label="Slow Solar Wind")
    ax[0].grid(True)
    ax[0].text(0.5, 0.9, f"Threat Score: {threat_score:.2f}", transform=ax[0].transAxes,
               fontsize=15, va="top", ha="center")
    ax[0].text(0.95, 0.95, "(a)", transform=ax[0].transAxes,
               fontsize=20, fontweight="bold", va="top", ha="left")
    ax[0].legend(loc="upper left")

    obs = df_plot["observed"].values
    pred = df_plot["predicted"].values
    mask_model  = ~np.isnan(obs) & ~np.isnan(pred)
    
    # --- Correlation plot
    ax[1].scatter(obs[mask_model], pred[mask_model], alpha=0.7, color="purple", label="Solar wind speed")

    slope, intercept = np.polyfit(obs[mask_model], pred[mask_model], 1)
    fitted_y = slope *obs[mask_model] + intercept
    ax[1].plot(obs[mask_model], fitted_y, color="black", linestyle="--", label="correlation")

    ax[1].set_title("Correlation Plot", fontsize=15, fontweight="bold")
    ax[1].set_xlabel("V(Km/s) Observed", fontsize=12, fontweight="bold")
    ax[1].set_ylabel("V(Km/s) Predicted", fontsize=12, fontweight="bold")
    ax[1].text(0.05, 0.95, f"r = {corr:.2f}\np = {p_value:.3e}",
               transform=ax[1].transAxes, fontsize=12,
               verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white"))
    ax[1].text(0.9, 0.95, "(b)", transform=ax[1].transAxes,
               fontsize=20, fontweight="bold", va="top", ha="left")
    ax[1].set_xlim(250, 750)
    ax[1].set_ylim(250, 750)
    ax[1].legend(loc="upper center")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ===============================
# Plot: Observed vs Model vs 27-day & N-day Persistence (+ Correlations)
# ===============================
def plot_results_with_persistence(sw_dates, y_obs, y_pred_model, y_pred_pers27, y_pred_persN,
                                  icme_dates, delay, fold_index,
                                  corr_model, p_model, corr_pers27, p_pers27,
                                  corr_persN, p_persN,
                                  save_path, title_prefix="SpeedNet",
                                  max_nans=5, gap_size=3, model_type="EUV"):
    """
    Four-line time series (Observed, Model, 27-day, N-day) with small markers
    + correlation panel for all three predictions.
    """

    # --- Make DataFrame ---
    df_plot = pd.DataFrame({
        "sw_date": sw_dates,
        "observed": y_obs.reshape(-1),
        "model": y_pred_model.reshape(-1),
        "pers27": y_pred_pers27.reshape(-1),
        "persN": y_pred_persN.reshape(-1)
    }).sort_values("sw_date").reset_index(drop=True)

    # --- Compress long NaN runs ---
    compressed_rows = []
    nan_count = 0
    for _, row in df_plot.iterrows():
        if pd.isna(row["observed"]) and pd.isna(row["model"]) and pd.isna(row["pers27"]) and pd.isna(row["persN"]):
            nan_count += 1
            if nan_count <= max_nans:
                compressed_rows.append(row)
        else:
            nan_count = 0
            compressed_rows.append(row)
    df_plot = pd.DataFrame(compressed_rows).reset_index(drop=True)

    # --- Build artificial index with spacing for gaps ---
    index_with_gaps = []
    current_idx = 0
    prev_date = None
    for d in df_plot["sw_date"]:
        if prev_date is not None and (d - prev_date).days > 1:
            current_idx += gap_size
        index_with_gaps.append(current_idx)
        current_idx += 1
        prev_date = d
    df_plot["plot_idx"] = index_with_gaps

    idx   = df_plot["plot_idx"].values
    dates = df_plot["sw_date"].values
    obs   = df_plot["observed"].values
    pred  = df_plot["model"].values
    pers27 = df_plot["pers27"].values
    persN  = df_plot["persN"].values

    # --- Helper: plot with gaps & small markers ---
    def plot_with_gaps(ax, x, y, color, label, marker, gap_tolerance=gap_size, linewidth=1.0, markersize=3):
        seg_x, seg_y = [], []
        for i in range(len(x)):
            if np.isnan(y[i]):
                if seg_x:
                    ax.plot(seg_x, seg_y, color=color, linewidth=linewidth,
                            marker=marker, markersize=markersize,
                            label=label if label else None)
                    label = None
                    seg_x, seg_y = [], []
                continue
            if seg_x and (x[i] - seg_x[-1]) > gap_tolerance:
                ax.plot(seg_x, seg_y, color=color, linewidth=linewidth,
                        marker=marker, markersize=markersize,
                        label=label if label else None)
                label = None
                seg_x, seg_y = [], []
            seg_x.append(x[i]); seg_y.append(y[i])
        if seg_x:
            ax.plot(seg_x, seg_y, color=color, linewidth=linewidth,
                    marker=marker, markersize=markersize,
                    label=label if label else None)

    # --- Figure ---
    fig = plt.figure(figsize=(32, 8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[8, 2.8], wspace=0.25)
    ax_ts   = fig.add_subplot(gs[0, 0])
    ax_corr = fig.add_subplot(gs[0, 1])

    plt.style.use("default")
    fig.suptitle(f"{title_prefix} SW Prediction vs Observed (Fold {fold_index}, Delay {delay})",
                 fontsize=16, fontweight="bold", y=1.02)

    # --- Time series (clean colors, thin lines, tiny markers) ---
    plot_with_gaps(ax_ts, idx, pred,   color="blue",   label=f"SpeedNet-{model_type}",      marker="o")
    plot_with_gaps(ax_ts, idx, pers27, color="red",    label="27-day",     marker="s")
    plot_with_gaps(ax_ts, idx, persN,  color="orange", label="N-day",      marker="^")
    plot_with_gaps(ax_ts, idx, obs,    color="green",  label="Observed",   marker=".")

    # ICME date tick marks
    for icme_date in icme_dates:
        if np.any(np.abs((dates - np.datetime64(icme_date)).astype("timedelta64[D]").astype(int)) <= 2):
            icme_idx = df_plot.loc[df_plot["sw_date"] == icme_date, "plot_idx"]
            if not icme_idx.empty:
                ax_ts.axvline(x=icme_idx.iloc[0], color="black",
                              linestyle="-", ymax=1, ymin=0.95, linewidth=1.0)

    ax_ts.set_ylabel("V (Km/s)", fontsize=12, fontweight="bold")
    ax_ts.set_xlabel("SW DATE", fontsize=12, fontweight="bold")
    ax_ts.grid(False)
    ax_ts.legend(loc="upper left")

    # Sparse x ticks with readable labels
    if len(dates) > 0:
        step = max(1, len(idx)//15)
        tick_locs = idx[::step]
        tick_labels = [pd.to_datetime(dates[i]).strftime("%d-%m-%Y") for i in range(0, len(dates), step)]
        ax_ts.set_xticks(tick_locs)
        ax_ts.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    # --- Correlation panel ---
    mask_model  = ~np.isnan(obs) & ~np.isnan(pred)
    mask_pers27 = ~np.isnan(obs) & ~np.isnan(pers27)
    mask_persN  = ~np.isnan(obs) & ~np.isnan(persN)

    ax_corr.scatter(obs[mask_model],  pred[mask_model],    alpha=0.6, label=f"Obs vs SpeedNet-{model_type}",  s=14, c="blue")
    ax_corr.scatter(obs[mask_pers27], pers27[mask_pers27], alpha=0.6, label="Obs vs 27-day", s=14, c="red")
    ax_corr.scatter(obs[mask_persN],  persN[mask_persN],   alpha=0.6, label="Obs vs N-day",  s=14, c="orange")

    # Optional simple fit lines
    if mask_model.sum() > 1:
        m_slope, m_inter = np.polyfit(obs[mask_model], pred[mask_model], 1)
        ax_corr.plot(obs[mask_model], m_slope*obs[mask_model] + m_inter, 'k--', linewidth=1.0, label=f"SpeedNet-{model_type} fit")
    if mask_pers27.sum() > 1:
        p27_slope, p27_inter = np.polyfit(obs[mask_pers27], pers27[mask_pers27], 1)
        ax_corr.plot(obs[mask_pers27], p27_slope*obs[mask_pers27] + p27_inter, 'r--', linewidth=1.0, label="27-day fit")
    if mask_persN.sum() > 1:
        pN_slope, pN_inter = np.polyfit(obs[mask_persN], persN[mask_persN], 1)
        ax_corr.plot(obs[mask_persN], pN_slope*obs[mask_persN] + pN_inter, 'orange', linestyle="--", linewidth=1.0, label="N-day fit")

    ax_corr.set_title("Correlation", fontsize=14, fontweight="bold")
    ax_corr.set_xlabel("Observed V (Km/s)", fontsize=12, fontweight="bold")
    ax_corr.set_ylabel("Predicted V (Km/s)", fontsize=12, fontweight="bold")

    ax_corr.text(0.98, 0.04,
                 f"SpeedNet-{model_type}: r={corr_model:.2f}, p={p_model:.2e}\n"
                 f"27-day: r={corr_pers27:.2f}, p={p_pers27:.2e}\n"
                 f"N-day: r={corr_persN:.2f}, p={p_persN:.2e}",
                 transform=ax_corr.transAxes, fontsize=12,
                 verticalalignment="bottom", horizontalalignment="right",
                 bbox=dict(boxstyle="round", facecolor="white"))

    ax_corr.legend(loc="upper left")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {save_path}")

# =================================
# MODEL OPERATION
# =================================

def save_data_csv(y_test, preds, std_test, df_test_matched, phase, fold_index, num_delay, save_path):
    """
    Save predictions with observed values and uncertainties to CSV.

    Parameters
    ----------
    y_test : np.ndarray
        Ground truth test labels.
    preds : np.ndarray
        Model predictions.
    std_test : np.ndarray
        Uncertainty values (std. dev. of observed).
    df_test_matched : pd.DataFrame
        DataFrame containing matched test data (must include 'DATE').
    phase : str
        Experiment identifier (used in filenames).
    fold_index : int
        Fold index used.
    num_delay : int
        Delay value used.
    save_path : str
        Directory where the CSV will be saved.
    """
    os.makedirs(save_path, exist_ok=True)

    data_pred = {
        'date': df_test_matched["sw_date"].values,
        'test_original': y_test,
        'test_predicted': preds,
        'std_original': std_test
    }
    pred_df = pd.DataFrame(data_pred)

    # Create complete date range from min to max
    pred_date_range = pd.date_range(start=pred_df['date'].min(), end=pred_df['date'].max())

    # Reindex to insert NaNs for missing dates
    pred_df = pred_df.set_index('date').reindex(pred_date_range).rename_axis('date').reset_index()

    filename = os.path.join(save_path, f'prediction_{phase}_fold{fold_index}_delay{num_delay}.csv')
    pred_df.to_csv(filename, index=False)
    print(f"Saved predictions CSV to {filename}")

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

def compute_metrics(y_true, y_pred, std_dev=None, hsemetric=None):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mae = MeanAbsoluteError()(y_true, y_pred).numpy()
    mse = MeanSquaredError()(y_true, y_pred).numpy()
    rmse = float(np.sqrt(mse))
    n = len(y_true)

    # --- Uncertainty of MSE
    errors = y_true - y_pred
    sq_errors = errors ** 2
    mse_std = np.std(sq_errors, ddof=1) / np.sqrt(n)

    # --- Uncertainty of RMSE (propagation)
    rmse_std = (0.5 / rmse) * mse_std if rmse > 0 else np.nan

    # guard for tiny arrays
    if len(y_true) > 1 and len(y_pred) > 1:
        corr, p_value = pearsonr(y_true, y_pred)
    else:
        corr, p_value = (np.nan, np.nan)
    
    if abs(corr) < 1 and n > 3:
        sigma_z = 1 / np.sqrt(n - 3)
        corr_std = (1 - corr**2) * sigma_z
    else:
        corr_std = np.nan
        
    # threat score
    
    if hsemetric is not None:
        try:
            tp, fp, fn = hsemetric.BatchwiseHSE(y_true, np.array(y_pred))
            threat_score = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
            bias = (tp + fn) / (tp + fp) if (tp + fp) > 0 else np.nan
            ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            far = fp / (tp + fp) if (tp + fp) > 0 else np.nan
        except Exception:
            threat_score = bias = ppv = far = np.nan
    else:
        threat_score = bias = ppv = far = np.nan

    if std_dev is not None:
        reduced_chi_sq = reduced_chi_squared(y_true, y_pred, std_dev)
    else:
        reduced_chi_sq = np.nan

    return dict(MAE=float(mae), MSE=float(mse), MSE_error = float(mse_std), RMSE=float(rmse), 
                RMSE_error = float(rmse_std), Corr=float(corr),
                Corr_std=float(corr_std), P_Value=float(p_value), Threat_Score=float(threat_score),
                Bias=float(bias), PPV=float(ppv), FAR=float(far), Reduced_Chi_Squared=float(reduced_chi_sq))


def compute_metrics_with_uncertainty(y_true, y_pred):
    n = len(y_true)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # --- Central values from Keras
    mae = MeanAbsoluteError()(y_true, y_pred).numpy()
    mse = MeanSquaredError()(y_true, y_pred).numpy()
    rmse = np.sqrt(mse)

    # --- Uncertainty of MSE
    errors = y_true - y_pred
    sq_errors = errors ** 2
    mse_std = np.std(sq_errors, ddof=1) / np.sqrt(n)

    # --- Uncertainty of RMSE (propagation)
    rmse_std = (0.5 / rmse) * mse_std if rmse > 0 else np.nan

    # --- Pearson correlation + uncertainty (Fisher z-transform)
    r, p_value = pearsonr(y_true, y_pred)
    if abs(r) < 1 and n > 3:
        sigma_z = 1 / np.sqrt(n - 3)
        corr_std = (1 - r**2) * sigma_z
    else:
        corr_std = np.nan

    return mae, mse, mse_std, rmse, rmse_std, r, corr_std, p_value


# ------------------------
# Persistence baselines
# ------------------------
def persistence_27_on_dates(df_sw, sw_dates, period_days=27.0, fill_method=None):
    """
    Classic persistence: predicted(t) = V(t - period_days)
    """
    s = (df_sw[["DATE", "speed"]]
         .dropna()
         .sort_values("DATE")
         .set_index("DATE")["speed"]
         .asfreq("1D"))
    if fill_method in ("ffill", "bfill"):
        s = getattr(s, fill_method)()

    lag = pd.Timedelta(days=period_days)
    preds = []
    for d in sw_dates:
        prev = d - lag
        val = float(s.loc[prev]) if prev in s.index else np.nan
        preds.append(val)
    return np.array(preds, dtype=float)


def persistence_N_on_dates(df_sw, sw_dates, delay, period_days=27.0, fill_method=None):
    """
    N-day persistence WITH delay:
        predicted(t) = V(t + delay - period_days)
    """
    s = (df_sw[["DATE", "speed"]]
         .dropna()
         .sort_values("DATE")
         .set_index("DATE")["speed"]
         .asfreq("1D"))
    if fill_method in ("ffill", "bfill"):
        s = getattr(s, fill_method)()

    lag = pd.Timedelta(days=period_days)
    preds = []
    for d in sw_dates:
        prev = d + timedelta(days=delay) - lag
        val = float(s.loc[prev]) if prev in s.index else np.nan
        preds.append(val)
    return np.array(preds, dtype=float)


def save_individual_attributions(X_test, X_test_standardized, sw_test_dates, delay, model, phase, base_save_dir):
    """
    Save individual attribution maps (Grad-CAM, Guided Grad-CAM, IG) as separate .npy files.
    Organized into subfolders by type.
    
    Args:
        X_test: Numpy array of test images
        sw_test_dates: List of dates or identifiers matching each sample
        delay: Integer or string to append to file name
        model: The trained model to use
        phase: Name of the phase (used for folder structure)
        base_save_dir: Root directory where results will be saved
    """
    x_map_dir = os.path.join(base_save_dir, "Attribution", "Input_Maps")
    if X_test_standardized is not None:
        x_standardized_dir = os.path.join(base_save_dir, "Attribution", "Input_Maps_Standardized")
        os.makedirs(x_standardized_dir, exist_ok=True)
    gradcam_dir = os.path.join(base_save_dir, "Attribution", "Gradcam")
    # guided_dir = os.path.join(base_save_dir, phase, "Attribution", "GuidedGradcam")
    ig_dir = os.path.join(base_save_dir, "Attribution", "IG")

    os.makedirs(x_map_dir, exist_ok=True)
    os.makedirs(gradcam_dir, exist_ok=True)
    # os.makedirs(guided_dir, exist_ok=True)
    os.makedirs(ig_dir, exist_ok=True)

    for i in range(len(X_test)):
        img = X_test[i].reshape(1, *X_test.shape[1:])
        # gradcam, guided_gradcam, ig, _ = explain_image_with_heatmaps(img, model)
        gradcam, ig, _ = explain_image_with_heatmaps(img, model)

        new_date = sw_test_dates[i] - timedelta(days=delay)

        # Format to YYYY-MM-DD
        new_date_str = new_date.strftime("%Y_%m_%d")
            
        filename = f"{new_date_str}.npy"

        np.save(os.path.join(x_map_dir, filename), X_test[i])
        if X_test_standardized is not None:
            np.save(os.path.join(x_standardized_dir, filename), X_test_standardized[i, :, :, 0])
        np.save(os.path.join(gradcam_dir, filename), gradcam)
        # np.save(os.path.join(guided_dir, filename), guided_gradcam)
        np.save(os.path.join(ig_dir, filename), ig)

        # Optional: cleanup per iteration
        K.clear_session()
        # gc.collect()

# -------------------------
# Main pipeline
# -------------------------

def train_pipeline(phase, sw_csv, icme_excel, img_path, std_csv, model_path,
                   test_start, test_end,
                   lstm_unit=70, dense_unit=300, batch_size=12, seed=42,
                   fold_index=None, delay=None, to_train=True,
                   obt_attrib=False, model_type=None,
                   standardized_path=None, save_path="./results",
                   hsemetric=None, num_splits=5,to_do_persistent=False,
                   persistence_period_days=27,
                   N_period_days=27,
                   persistence_fill="ffill",
                   all_metrics=None):
    
    """
    Train or evaluate SpeedNet models for solar wind speed prediction with attribution support.

    This pipeline can be run in two main modes:
    1. Training Mode (`to_train=True`):
       - Splits training/validation/test sets.
       - Trains models, evaluates them, and saves performance metrics into Excel.
    2. Evaluation/Inference Mode (`to_train=False`):
       - Skips training and loads existing trained models.
       - Produces predictions on the fixed test set.
       - Saves predictions as a CSV file.
       - Optionally computes attribution maps.

    All outputs (metrics, predictions, plots, npy files) are saved under `save_path`.

    Parameters
    ----------
    phase : str
        Experiment identifier (used in filenames).
    sw_csv : str
        Path to solar wind CSV.
    icme_excel : str
        Path to ICME Excel file.
    img_path : str
        Path to EUV images.
    std_csv : str
        Path to standard deviation CSV.
    model_path : str
        Path to save models/checkpoints.
    test_start, test_end : str or datetime-like
        Start/end of test set.
    lstm_unit, dense_unit, batch_size : int
        Model/training hyperparameters.
    seed : int
        Random seed.
    fold_index : int or None
        If None, run all folds. If int, run only that fold.
    delay : int or None
        If None, run all delays. If int, run only that delay.
    to_train : bool
        If True: train models and return metrics DataFrame.
        If False: skip training, only test/predict.
    obt_attrib : bool
        If True, compute attribution maps.
    model_type : str or None
        One of (None, "EUV", "BM").
    standardized_path : str
        Required if model_type="SpeedNet_BM".
    save_path : str
        Directory to save outputs (CSV, npy, plots, metrics).
    hsemetric : object or None
        If not None, must provide a `BatchwiseHSE(y_true, y_pred)` method to compute
        (TP, FP, FN). Used for threat score, bias, ppv, far.
        to_do_persistent : bool, default=False
        If True, evaluates persistence baselines (27-day and N-day).
    persistence_period_days : int, default=27
        Period for 27-day persistence baseline.
    N_period_days : int, default=1
        Period for N-day persistence baseline.
    persistence_fill : {"ffill", "bfill"}, default="ffill"
        Fill method for persistence baselines.
    all_metrics : list, optional
        Existing metrics list to extend across folds/delays.

    Returns
    -------
    If `to_train=True`:
        pd.DataFrame with metrics for each fold and delay.
    If `to_train=False`:
        tuple: (y_test, preds, [df_test_matched], icme_dates, delay, fold_index).
    """
  
    os.makedirs(save_path, exist_ok=True)

    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Prepare dataset (user-defined)
    df, img_dict, icme_dates = prepare_dataset(sw_csv, icme_excel, img_path, std_csv)

    # Train/Test split
    
    test_mask = (df["DATE"] >= pd.to_datetime(test_start)) & (df["DATE"] <= pd.to_datetime(test_end))
    
    df_test = df[test_mask].reset_index(drop=True)
    df_train = df[~test_mask].reset_index(drop=True)

    

    delays = [delay] if delay is not None else [1, 2, 3, 4, 5]
    all_metrics = []
    
    if all_metrics is None:
        all_metrics = []

    for d in delays:
        print(f"\n=== Delay {d} ===")
        df_train_matched = match_data(df_train, img_dict, d)
        df_test_matched = match_data(df_test, img_dict, d)

        if df_test_matched.empty or df_train_matched.empty:
            print("Skipping delay due to no matches")
            continue
        
        print(f"Train size: {len(df_train_matched)}, Test size: {len(df_test_matched)}")
        X_test = np.array([load_map(os.path.join(img_path, f)) for f in df_test_matched["img_file"]])
        if standardized_path is not None:
            X_test_standardized = np.array([load_map(os.path.join(standardized_path, f)) for f in df_test_matched["img_file"]])
            
        y_test = df_test_matched["speed"].values
        std_test = df_test_matched["std_speed"].values
        
        sw_test_dates = pd.to_datetime(df_test_matched["sw_date"].values)
        
        df_sw = df.copy()

        # ----------------
        # Training mode
        # ----------------
        if to_train:
            num_samples = len(df_train_matched)
            fold_size = num_samples // num_splits

            for fold_id in range(num_splits):
                val_start = fold_id * fold_size
                val_end = min(val_start + fold_size, num_samples)

                if (fold_index is not None) and (fold_id != fold_index):
                    continue

                print(f"\n--- Fold {fold_id} ---")
                val_df = df_train_matched.iloc[val_start:val_end]
                train_df = pd.concat([df_train_matched.iloc[:val_start], df_train_matched.iloc[val_end:]])

                X_train = np.array([load_map(os.path.join(img_path, f)) for f in train_df["img_file"]])
                y_train = train_df["speed"].values
                X_val = np.array([load_map(os.path.join(img_path, f)) for f in val_df["img_file"]])
                y_val = val_df["speed"].values

                input_shape = X_train.shape[1:]
                model = SpeedNet(input_shape, lstm_unit, dense_unit)
                model.compile(optimizer=Adam(1e-4), loss="mse", metrics=["mae", "mse"])

                model_filename = os.path.join(model_path, f"{phase}_best_model_fold{fold_id}_delay{d}.keras")
                checkpoint = ModelCheckpoint(model_filename, monitor="val_loss",
                                             save_best_only=True, mode="min", verbose=1)

                model.fit(X_train, y_train,
                          validation_data=(X_val, y_val),
                          epochs=100, batch_size=batch_size,
                          callbacks=[checkpoint], verbose=1)

                loaded = load_model(model_filename)
                preds_model = loaded.predict(X_test, verbose=0).reshape(-1)

                
                # --- Metrics for MODEL ---
                # Save predictions
                save_data_csv(y_test, preds_model, std_test, df_test_matched,
                            phase, fold_id or 0, d, save_path)

                reduced_chi_sq_speednet = reduced_chi_squared(y_test, preds_model, std_test)
                mae, mse, mse_std, rmse, rmse_std, corr_m, corr_std_m, p_m = compute_metrics_with_uncertainty(y_test, preds_model)
                
                if hsemetric is not None:
                    try:
                        tp, fp, fn = hsemetric.BatchwiseHSE(y_test, np.array(preds_model))
                        threat_score = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
                        bias = (tp + fn) / (tp + fp) if (tp + fp) > 0 else np.nan
                        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
                        far = fp / (tp + fp) if (tp + fp) > 0 else np.nan
                    except Exception:
                        threat_score = bias = ppv = far = np.nan
                else:
                    threat_score = bias = ppv = far = np.nan
                       
                plot_file_without_per = os.path.join(save_path, f"results_eval_fold{fold_id}_delay{d}_{model_type}.png")
                plot_results(y_test, preds_model, df_test_matched, 
                         icme_dates, d, fold_id,
                         threat_score, corr_m, p_m, 
                         plot_file_without_per, model_type)
                

                row = (fold_id, d, model_type,
                    f"{rmse:.3f} ± {rmse_std:.3f}",
                    f"{corr_m:.3f} ± {corr_std_m:.3f}",
                    f"{mae:.3f}",
                    f"{mse:.3f} ± {mse_std:.3f}",
                    p_m,
                    threat_score, bias, ppv, far, 
                    reduced_chi_sq_speednet,
                    batch_size, lstm_unit, dense_unit)
                
                all_metrics.append(row)

                # Persistence baselines
                preds_p27, preds_pN = None, None
                if to_do_persistent:
                    preds_p27 = persistence_27_on_dates(df_sw, sw_test_dates,
                                                        period_days=persistence_period_days,
                                                        fill_method=persistence_fill)
                    mask_valid_p27 = ~np.isnan(preds_p27)
                    y_obs_p27 = y_test[mask_valid_p27]
                    std_obs_p27 = std_test[mask_valid_p27]
                    y_p27     = preds_p27[mask_valid_p27]

                    reduced_chi_sq_p27 = reduced_chi_squared(y_obs_p27, y_p27, std_obs_p27)
                    mae_p27, mse_p27, mse_std_p27, rmse_p27, rmse_std_p27, corr_p27, corr_std_p27, p_p27 = compute_metrics_with_uncertainty(y_obs_p27, y_p27)
                        
                    if hsemetric is not None:
                        try:
                            tp_p27, fp_p27, fn_p27 = hsemetric.BatchwiseHSE(y_obs_p27, np.array(y_p27))
                            threat_score_p27 = tp_p27 / (tp_p27 + fp_p27 + fn_p27) if (tp_p27 + fp_p27 + fn_p27) > 0 else np.nan
                            bias_p27 = (tp_p27 + fn_p27) / (tp_p27 + fp_p27) if (tp_p27 + fp_p27) > 0 else np.nan
                            ppv_p27 = tp_p27 / (tp_p27 + fp_p27) if (tp_p27 + fp_p27) > 0 else np.nan
                            far_p27 = fp_p27 / (tp_p27 + fp_p27) if (tp_p27 + fp_p27) > 0 else np.nan
                        except Exception:
                            threat_score_p27 = bias_p27 = ppv_p27 = far_p27 = np.nan
                    else:
                        threat_score_p27 = bias_p27 = ppv_p27 = far_p27 = np.nan
                    
                    row2 = (fold_id, d, "Persistence-27",
                    f"{rmse_p27:.3f} ± {rmse_std_p27:.3f}",
                    f"{corr_p27:.3f} ± {corr_std_p27:.3f}",
                    f"{mae_p27:.3f}",
                    f"{mse_p27:.3f} ± {mse_std_p27:.3f}",
                    p_p27,
                    threat_score_p27, bias_p27, ppv_p27, far_p27,
                    reduced_chi_sq_p27,
                    batch_size, lstm_unit, dense_unit)
                    
                    all_metrics.append(row2)
                    
                    preds_pN = persistence_N_on_dates(df_sw, sw_test_dates,
                                                    delay=d,
                                                    period_days=N_period_days,
                                                    fill_method=persistence_fill)
                    mask_valid_pN = ~np.isnan(preds_pN)
                    y_obs_pN = y_test[mask_valid_pN]
                    std_obs_pN = std_test[mask_valid_pN]
                    y_pN     = preds_pN[mask_valid_pN]

                    reduced_chi_sq_pN = reduced_chi_squared(y_obs_pN, y_pN, std_obs_pN)
                    mae_pN, mse_pN, mse_std_pN, rmse_pN, rmse_std_pN, corr_pN, corr_std_pN, p_pN = compute_metrics_with_uncertainty(y_obs_pN, y_pN)

                    if hsemetric is not None:
                        try:
                            tp_pN, fp_pN, fn_pN = hsemetric.BatchwiseHSE(y_obs_pN, np.array(y_pN))
                            threat_score_pN = tp_pN / (tp_pN + fp_pN + fn_pN) if (tp_pN + fp_pN + fn_pN) > 0 else np.nan
                            bias_pN = (tp_pN + fn_pN) / (tp_pN + fp_pN) if (tp_pN + fp_pN) > 0 else np.nan
                            ppv_pN = tp_pN / (tp_pN + fp_pN) if (tp_pN + fp_pN) > 0 else np.nan
                            far_pN = fp_pN / (tp_pN + fp_pN) if (tp_pN + fp_pN) > 0 else np.nan
                        except Exception:
                            threat_score_pN = bias_pN = ppv_pN = far_pN = np.nan
                    else:
                        threat_score_pN = bias_pN = ppv_pN = far_pN = np.nan
                    
                    # --- Persistence-N row ---
                    row3 = (fold_id, d, f"Persistence-{N_period_days}",
                            f"{rmse_pN:.3f} ± {rmse_std_pN:.3f}",
                            f"{corr_pN:.3f} ± {corr_std_pN:.3f}",
                            f"{mae_pN:.3f}",
                            f"{mse_pN:.3f} ± {mse_std_pN:.3f}",
                            p_pN,
                            threat_score_pN, bias_pN, ppv_pN, far_pN,
                            reduced_chi_sq_pN,
                            batch_size, lstm_unit, dense_unit)
                    
                    all_metrics.append(row3)
                    persis_csv_fil = os.path.join(save_path, f"27_day_persistence_fold")
                    save_data_csv(y_test, preds_p27, std_test, df_test_matched,
                                  phase, fold_id or 0, d, persis_csv_fil)
                    
                    N_persis_csv_file = os.path.join(save_path, f"N_persistence_fold")
                    save_data_csv(y_test, preds_pN, std_test, df_test_matched,
                                  phase, fold_id or 0, d, N_persis_csv_file)
                    
                    plot_file = os.path.join(save_path, f"results_eval_fold{fold_id}_delay{d}_persistence.png")
                    plot_results_with_persistence(sw_test_dates, y_test, preds_model,
                                                preds_p27, preds_pN,
                                                icme_dates, d, fold_id,
                                                corr_m, p_m, corr_p27, p_p27, corr_pN, p_pN,
                                                save_path=plot_file,
                                                title_prefix="SpeedNet", model_type=model_type)

                    

        # ----------------
        # Evaluation mode
        # ----------------
        else:
            model_filename = os.path.join(model_path, f"{phase}_best_model_fold{fold_index}_delay{d}.keras")
            loaded = load_model(model_filename)
            preds_model = loaded.predict(X_test, verbose=0).reshape(-1)

            # Save predictions
            save_data_csv(y_test, preds_model, std_test, df_test_matched,
                        phase, fold_index or 0, d, save_path)
            reduced_chi_sq_speednet = reduced_chi_squared(y_test, preds_model, std_test)
            mae, mse, mse_std, rmse, rmse_std, corr_m, corr_std_m, p_m = compute_metrics_with_uncertainty(y_test, preds_model)
                    
            if hsemetric is not None:
                try:
                    tp, fp, fn = hsemetric.BatchwiseHSE(y_test, np.array(preds_model))
                    threat_score = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
                    bias = (tp + fn) / (tp + fp) if (tp + fp) > 0 else np.nan
                    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
                    far = fp / (tp + fp) if (tp + fp) > 0 else np.nan
                except Exception:
                    threat_score = bias = ppv = far = np.nan
            else:
                threat_score = bias = ppv = far = np.nan
            
            plot_file_without_per = os.path.join(save_path, f"results_eval_fold{fold_index}_delay{d}_{model_type}.png")
            
            plot_results(y_test, preds_model, df_test_matched, 
                             icme_dates, d, fold_index,
                             threat_score, corr_m, p_m, 
                             plot_file_without_per, model_type)   
                
            row = (fold_index, d, model_type,
                f"{rmse:.3f} ± {rmse_std:.3f}",
                f"{corr_m:.3f} ± {corr_std_m:.3f}",
                f"{mae:.3f}",
                f"{mse:.3f} ± {mse_std:.3f}",
                p_m,
                threat_score, bias, ppv, far,
                reduced_chi_sq_speednet,
                batch_size, lstm_unit, dense_unit)
            all_metrics.append(row)

            # Persistence baselines
            preds_p27, preds_pN = None, None
            if to_do_persistent:
                preds_p27 = persistence_27_on_dates(df_sw, sw_test_dates,
                                                    period_days=persistence_period_days,
                                                    fill_method=persistence_fill)
                mask_valid_p27 = ~np.isnan(preds_p27)
                y_obs_p27 = y_test[mask_valid_p27]
                std_obs_p27 = std_test[mask_valid_p27]
                y_p27     = preds_p27[mask_valid_p27]
                
                reduced_chi_sq_p27 = reduced_chi_squared(y_obs_p27, y_p27, std_obs_p27)
                mae_p27, mse_p27, mse_std_p27, rmse_p27, rmse_std_p27, corr_p27, corr_std_p27, p_p27 = compute_metrics_with_uncertainty(y_obs_p27, y_p27)
                     
                if hsemetric is not None:
                    try:
                        tp_p27, fp_p27, fn_p27 = hsemetric.BatchwiseHSE(y_obs_p27, np.array(y_p27))
                        threat_score_p27 = tp_p27 / (tp_p27 + fp_p27 + fn_p27) if (tp_p27 + fp_p27 + fn_p27) > 0 else np.nan
                        bias_p27 = (tp_p27 + fn_p27) / (tp_p27 + fp_p27) if (tp_p27 + fp_p27) > 0 else np.nan
                        ppv_p27 = tp_p27 / (tp_p27 + fp_p27) if (tp_p27 + fp_p27) > 0 else np.nan
                        far_p27 = fp_p27 / (tp_p27 + fp_p27) if (tp_p27 + fp_p27) > 0 else np.nan
                    except Exception:
                        threat_score_p27 = bias_p27 = ppv_p27 = far_p27 = np.nan
                else:
                    threat_score_p27 = bias_p27 = ppv_p27 = far_p27 = np.nan
                
                row2 = (fold_index, d, "Persistence-27",
                f"{rmse_p27:.3f} ± {rmse_std_p27:.3f}",
                f"{corr_p27:.3f} ± {corr_std_p27:.3f}",
                f"{mae_p27:.3f}",
                f"{mse_p27:.3f} ± {mse_std_p27:.3f}",
                p_p27,
                threat_score_p27, bias_p27, ppv_p27, far_p27,
                reduced_chi_sq_p27,
                batch_size, lstm_unit, dense_unit)
                
                all_metrics.append(row2)
                
                preds_pN = persistence_N_on_dates(df_sw, sw_test_dates,
                                                delay=d,
                                                period_days=N_period_days,
                                                fill_method=persistence_fill)
                mask_valid_pN = ~np.isnan(preds_pN)
                y_obs_pN = y_test[mask_valid_pN]
                std_obs_pN = std_test[mask_valid_pN]
                y_pN     = preds_pN[mask_valid_pN]

                reduced_chi_sq_pN = reduced_chi_squared(y_obs_pN, y_pN, std_obs_pN)
                mae_pN, mse_pN, mse_std_pN, rmse_pN, rmse_std_pN, corr_pN, corr_std_pN, p_pN = compute_metrics_with_uncertainty(y_obs_pN, y_pN)
                    
                if hsemetric is not None:
                    try:
                        tp_pN, fp_pN, fn_pN = hsemetric.BatchwiseHSE(y_obs_pN, np.array(y_pN))
                        threat_score_pN = tp_pN / (tp_pN + fp_pN + fn_pN) if (tp_pN + fp_pN + fn_pN) > 0 else np.nan
                        bias_pN = (tp_pN + fn_pN) / (tp_pN + fp_pN) if (tp_pN + fp_pN) > 0 else np.nan
                        ppv_pN = tp_pN / (tp_pN + fp_pN) if (tp_pN + fp_pN) > 0 else np.nan
                        far_pN = fp_pN / (tp_pN + fp_pN) if (tp_pN + fp_pN) > 0 else np.nan
                    except Exception:
                        threat_score_pN = bias_pN = ppv_pN = far_pN = np.nan
                else:
                    threat_score_pN = bias_pN = ppv_pN = far_pN = np.nan
                
                # --- Persistence-N row ---
                row3 = (fold_index, d, f"Persistence-{N_period_days}",
                        f"{rmse_pN:.3f} ± {rmse_std_pN:.3f}",
                        f"{corr_pN:.3f} ± {corr_std_pN:.3f}",
                        f"{mae_pN:.3f}",
                        f"{mse_pN:.3f} ± {mse_std_pN:.3f}",
                        p_pN,
                        threat_score_pN, bias_pN, ppv_pN, far_pN,
                        reduced_chi_sq_pN,
                        batch_size, lstm_unit, dense_unit)
                
                persis_csv_fil = os.path.join(save_path, f"27_day_persistence_fold")
                save_data_csv(y_test, preds_p27, std_test, df_test_matched,
                                  phase, fold_index or 0, d, persis_csv_fil)
                    
                N_persis_csv_file = os.path.join(save_path, f"N_persistence_fold")
                save_data_csv(y_test, preds_pN, std_test, df_test_matched,
                                  phase, fold_index or 0, d, N_persis_csv_file)
                    
                
                all_metrics.append(row3)
                
                plot_file = os.path.join(save_path, f"results_eval_fold{fold_index}_delay{d}_persistence.png")
                plot_results_with_persistence(sw_test_dates, y_test, preds_model,
                                            preds_p27, preds_pN,
                                            icme_dates, d, fold_index,
                                            corr_m, p_m, corr_p27, p_p27, corr_pN, p_pN,
                                            save_path=plot_file,
                                            title_prefix="SpeedNet", model_type=model_type)

            # Save attributions if required
            phase_start_date = pd.to_datetime(test_start)
            phase_start_year = phase_start_date.year

            phase_end_date = pd.to_datetime(test_end)
            phase_end_year = phase_end_date.year
            
            if obt_attrib and model_type is not None:
                if model_type == "EUV":
                    np.save(os.path.join(save_path, f'{phase_start_year}_{phase_end_year}_{phase}_standardized_SpeedNet_{model_type}.npy'), X_test)

                elif model_type == "BM":
                    if standardized_path is None:
                        raise ValueError("standardized_path is required for SpeedNet_BM")

                    np.save(os.path.join(save_path, f'{phase_start_year}_{phase_end_year}_{phase}_standardized_193_for_SpeedNet_{model_type}.npy'),
                            X_test_standardized[:, :, :, 0])
                    np.save(os.path.join(save_path, f'{phase_start_year}_{phase_end_year}_{phase}_binary_map.npy'), X_test)
                print("Computing attribution maps...")
                print("==============================")
                
                save_individual_attributions(
                    X_test=X_test,
                    X_test_standardized=X_test_standardized if model_type=="BM" else None,
                    sw_test_dates=sw_test_dates,
                    delay=d,
                    model=loaded,
                    phase=phase,
                    base_save_dir=save_path
                    )

                gradcam_maps, ig_maps = [], []
                for i in range(len(X_test)):
                    img = X_test[i].reshape(1, *X_test.shape[1:])
                    gradcam, ig, pred_value = explain_image_with_heatmaps(img, loaded)
                    gradcam_maps.append(gradcam)
                    ig_maps.append(ig)
                    tf.keras.backend.clear_session()

                np.save(os.path.join(save_path, f'{phase_start_year}_{phase_end_year}_{phase}_gradcam_maps.npy'), np.array(gradcam_maps))
                np.save(os.path.join(save_path, f'{phase_start_year}_{phase_end_year}_{phase}_integrated_gradients.npy'), np.array(ig_maps))

                
            # Return results
            if to_do_persistent:
                return y_test, preds_model, preds_p27, preds_pN, df_test_matched, icme_dates, d, fold_index
            else:
                return y_test, preds_model, df_test_matched, icme_dates, d, fold_index
        
        # Save final metrics
    if to_train and all_metrics:
        df_metrics = pd.DataFrame(all_metrics,
                        columns=["Fold", "Delay", "ModelType",
                                "RMSE", "Corr", "MAE", "MSE", "P_Value",
                                "Threat_Score", "Bias", "PPV", "FAR",
                                "Reduced_Chi_Sq",
                                "Batchsize", "LSTM", "Dense"])

        metrics_file = os.path.join(save_path, f"{phase}_full_metrics.xlsx")
        df_metrics.to_excel(metrics_file, index=False)
        print(f"Saved metrics to {metrics_file}")
        return df_metrics
    
# # ===============================
# Attribution FUNCTIONS
# ===============================

@tf.custom_gradient
def guided_relu(x):
    def grad(dy):
        return tf.cast(dy > 0, tf.float32) * tf.cast(x > 0, tf.float32) * dy
    return tf.nn.relu(x), grad


def replace_relu_with_guided(model):
    def convert(layer):
        if hasattr(layer, "activation") and layer.activation == tf.keras.activations.relu:
            config = layer.get_config()
            config["activation"] = guided_relu
            return layer.__class__.from_config(config)
        return layer

    guided_model = tf.keras.models.clone_model(model, clone_function=convert)
    guided_model.set_weights(model.get_weights())
    return guided_model


def explain_image_with_heatmaps(img_tensor, best_model, last_conv_layer_name="max_pooling2d_3", class_index=None, baseline=None):
    if isinstance(img_tensor, np.ndarray):
        img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)
    if len(img_tensor.shape) == 3:
        img_tensor = tf.expand_dims(img_tensor, axis=0)

    # -------- GRAD-CAM --------
    grad_model = Model(
        inputs=best_model.inputs,
        outputs=[best_model.get_layer(last_conv_layer_name).output, best_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]  # Shape: (H, W, C)
    conv_outputs = conv_outputs[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    conv_outputs = conv_outputs * pooled_grads
    heatmap = tf.reduce_sum(conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0).numpy()
    heatmap = cv2.resize(heatmap, (img_tensor.shape[2], img_tensor.shape[1]))

    # -------- GUIDED GRAD-CAM --------
    # guided_model = replace_relu_with_guided(best_model)

    # with tf.GradientTape() as tape:
    #     tape.watch(img_tensor)
    #     out = guided_model(img_tensor)
    #     loss = out[:, class_index]

    # guided_grads = tape.gradient(loss, img_tensor)[0].numpy()
    # guided_grads = cv2.resize(guided_grads, (img_tensor.shape[2], img_tensor.shape[1]))
    # guided_gradcam = guided_grads * heatmap[..., np.newaxis]
    

    # -------- INTEGRATED GRADIENTS --------
    def interpolate_images(baseline, image, alphas):
        return baseline + alphas * (image - baseline)

    def compute_integrated_gradients(image, baseline, model, class_index, steps=50):
        alphas = tf.linspace(0.0, 1.0, steps)
        alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
        interpolated = interpolate_images(baseline, image, alphas_x)
        interpolated = tf.reshape(interpolated, (steps,) + image.shape[1:])
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = model(interpolated)
            selected_output = pred[:, class_index]
        grads = tape.gradient(selected_output, interpolated)
        avg_grads = tf.reduce_mean(grads, axis=0)
        integrated_gradients = (image - baseline)[0] * avg_grads
        return integrated_gradients.numpy()

    ### I CHANGE THE BASELINE FROM JUST ZERO-LIKE
    # if baseline is None:
    #     baseline = tf.zeros_like(img_tensor)
    # integrated_grad = compute_integrated_gradients(img_tensor, baseline, best_model, int(class_index))
    
    ### TO RANDOM RESAMPLING
    if baseline is None:
        # Create shuffled baseline
        flat = img_tensor.numpy().reshape(-1, img_tensor.shape[-1])  # flatten [num_pixels, channels]
        np.random.shuffle(flat)  # shuffle pixel rows
        shuffled_img = flat.reshape(img_tensor.shape)
        baseline = tf.convert_to_tensor(shuffled_img, dtype=img_tensor.dtype)

    # Compute integrated gradients
    integrated_grad = compute_integrated_gradients(img_tensor, baseline, best_model, int(class_index))

    # return heatmap, guided_gradcam, integrated_grad, predictions.numpy()
    return heatmap, integrated_grad, predictions.numpy()
    
