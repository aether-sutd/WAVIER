# === Standard Library ===
import os
import sys
import warnings
from pathlib import Path
from random import randint

# === Third-party Libraries ===
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, LeaveOneGroupOut, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from tslearn.utils import to_time_series_dataset
from scipy.interpolate import Akima1DInterpolator, interp1d
import pywt
import plotly
from plotly.graph_objs import *

# === Local Imports ===
from inputs.task import get_tasks
from labels import TLXLabeller, TaskLabeller
from datasets import ExperimentDataset
from features import TemporalGazeFeatureExtractor, TemporalPupilFeatureExtractor

def plot_clustering(z_run, labels, engine='plotly', download=False, folder_name='clustering', perplexity=15):
    """
    Plot latent variables with color representing label values as a gradient.
    Colorbar shows the original TLX values (0-100).
    """
    def plot_clustering_plotly(z_run, labels):
        z_run_pca = TruncatedSVD(n_components=3).fit_transform(z_run)
        z_run_tsne = TSNE(perplexity=perplexity, min_grad_norm=1E-12, max_iter=3000).fit_transform(z_run)

        # PCA Plot
        trace = Scatter(
            x=z_run_pca[:, 0],
            y=z_run_pca[:, 1],
            mode='markers',
            marker=dict(
                color=labels,         # original values
                colorscale='coolwarm',    # red → blue
                colorbar=dict(title='TLX Score'),
                showscale=True
            )
        )
        fig = Figure(data=[trace], layout=Layout(title='PCA on z_run', showlegend=False))
        plotly.offline.iplot(fig)

        # t-SNE Plot
        trace = Scatter(
            x=z_run_tsne[:, 0],
            y=z_run_tsne[:, 1],
            mode='markers',
            marker=dict(
                color=labels,
                colorscale='coolwarm',
                colorbar=dict(title='TLX Score'),
                showscale=True
            )
        )
        fig = Figure(data=[trace], layout=Layout(title='tSNE on z_run', showlegend=False))
        plotly.offline.iplot(fig)

    def plot_clustering_matplotlib(z_run, labels, download, folder_name):
        z_run_pca = TruncatedSVD(n_components=3).fit_transform(z_run)
        z_run_tsne = TSNE(perplexity=perplexity, min_grad_norm=1E-12, max_iter=3000).fit_transform(z_run)

        # PCA Plot
        plt.scatter(z_run_pca[:, 0], z_run_pca[:, 1], c=labels, cmap='RdBu', marker='*', linewidths=0)
        plt.title('PCA on z_run')
        cbar = plt.colorbar()
        cbar.set_label('TLX Score')  # show original values
        if download:
            os.makedirs(folder_name, exist_ok=True)
            plt.savefig(os.path.join(folder_name, "pca.png"))
        else:
            plt.show()

        # t-SNE Plot
        plt.scatter(z_run_tsne[:, 0], z_run_tsne[:, 1], c=labels, cmap='RdBu', marker='*', linewidths=0)
        plt.title('tSNE on z_run')
        cbar = plt.colorbar()
        cbar.set_label('TLX Score')
        if download:
            os.makedirs(folder_name, exist_ok=True)
            plt.savefig(os.path.join(folder_name, "tsne.png"))
        else:
            plt.show()

    # Calculate R² score: how well latent space encodes continuous labels
    reg = LinearRegression()
    reg.fit(z_run, labels.reshape(-1, 1))
    labels_pred = reg.predict(z_run)
    r2 = r2_score(labels, labels_pred)
    print(f"Latent space R² w.r.t TLX: {r2:.4f}")

    # Choose engine
    if (download == False) & (engine == 'plotly'):
        plot_clustering_plotly(z_run, labels)
    elif (download) & (engine == 'plotly'):
        print("Can't download Plotly plots")
    elif engine == 'matplotlib':
        plot_clustering_matplotlib(z_run, labels, download, folder_name)

def akima_interpolate_zeros(X, pad_value=np.nan):
    """
    Interpolates zeros via Akima for ragged arrays (object dtype).
    Returns a new array with the same ragged structure and padding.
    """
    X_interp = []
    for sequence in X:
        seq_interp = np.copy(sequence)
        for feature_idx in range(sequence.shape[1]):
            signal = sequence[:, feature_idx]
            if np.isnan(pad_value):
                valid_idx = ~np.isnan(signal)
            else:
                valid_idx = signal != pad_value
            signal_unpadded = signal[valid_idx]
            x = np.arange(len(signal_unpadded))

            interp_signal = signal_unpadded.copy()
            zero_mask = interp_signal == 0
            valid = ~zero_mask
            x_valid = x[valid]
            y_valid = interp_signal[valid]

            if len(x_valid) > 2:
                interpolator = Akima1DInterpolator(x_valid, y_valid)
                interp_signal[zero_mask] = interpolator(x[zero_mask])

            # Put back into padded array
            signal_interp_padded = signal.copy()
            signal_interp_padded[valid_idx] = interp_signal
            seq_interp[:, feature_idx] = signal_interp_padded
        X_interp.append(seq_interp)
    return np.array(X_interp, dtype='O')

def plot_interpolation_comparison(X_orig, X_interp, sample_idx=0, feature_idx=0, pad_value=np.nan):
    """
    Extracts original and interpolated signals for a sample/feature and plots them side by side.
    """
    orig_sequence = X_orig[sample_idx]
    interp_sequence = X_interp[sample_idx]
    orig_signal = orig_sequence[:, feature_idx]
    interp_signal = interp_sequence[:, feature_idx]
    if np.isnan(pad_value):
        valid_idx = ~np.isnan(orig_signal)
    else:
        valid_idx = orig_signal != pad_value
    orig_unpadded = orig_signal[valid_idx]
    interp_unpadded = interp_signal[valid_idx]
    x = np.arange(len(orig_unpadded))

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    y_min = min(np.nanmin(orig_unpadded), np.nanmin(interp_unpadded))
    y_max = max(np.nanmax(orig_unpadded), np.nanmax(interp_unpadded))

    axs[0].plot(x, orig_unpadded, color='tab:blue', label='Original')
    axs[0].set_title(f'Original Signal (Feature {feature_idx})')
    axs[0].set_xlabel('Time Steps')
    axs[0].set_ylabel('Signal Amplitude')
    axs[0].set_ylim(y_min, y_max)
    axs[0].legend()

    axs[1].plot(x, interp_unpadded, color='tab:orange', label='Processed')
    axs[1].set_title(f'Processed Signal (Feature {feature_idx})')
    axs[1].set_xlabel('Time Steps')
    axs[1].set_ylabel('Signal Amplitude')
    axs[1].set_ylim(y_min, y_max)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def plot_class_distribution(y_train, y_val):
    """
    Plot class distribution for combined train and validation labels.
    """
    import seaborn as sns
    # Combine train and validation labels
    y_all = np.concatenate([y_train, y_val])

    # Count occurrences of each class
    classes, counts = np.unique(y_all, return_counts=True)

    plt.figure(figsize=(10, 5))
    sns.countplot(x=y_all, hue=y_all, palette='pastel', edgecolor='black', legend=False)
    plt.xlabel('Class Label')
    plt.ylabel('Count')
    plt.title('Combined Class Distribution (Train + Validation)')
    plt.show()

def plot_sequence_length_hist(data, bins=30, title='Histogram of Sequence Lengths', label='Sequence'):
    """
    Plot a histogram of sequence lengths for a given dataset.
    
    Args:
        data: np.ndarray, shape (n_samples, seq_len, n_features)
        bins: int, number of histogram bins
        title: str, plot title
        label: str, legend label
    """
    import numpy as np
    import matplotlib.pyplot as plt

    lengths = [np.sum(~np.isnan(seq).any(axis=1)) for seq in data]
    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=bins, alpha=0.7, label=label)
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    plt.show()
    
    
    
def plot_dwt_coeff_length_hist(data, wavelet='db4', level=4, bins=30, title=None, label='DWT cA'):
    """
    Plot a histogram of DWT approximation coefficient lengths for a dataset.
    """
    from models.dwt_vae import sample_to_dwt_approx_only
    approx_lens = []
    for seq in data:
        seq_clean = seq[~np.isnan(seq).any(axis=1)]
        approx_parts, meta = sample_to_dwt_approx_only(seq_clean, wavelet=wavelet, level=level)
        for cA in approx_parts:
            approx_lens.append(len(cA))
    if title is None:
        title = f'Histogram of DWT cA Lengths (wavelet={wavelet}, level={level})'
    plt.figure(figsize=(8, 4))
    plt.hist(approx_lens, bins=bins, alpha=0.7, label=label)
    plt.xlabel('DWT Approximation Coefficient Length')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_history(hist):
    """Plots traning loss history across epochs."""
    plt.figure(figsize=(8, 5))
    plt.plot(hist)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VRAE Training Loss Across Epochs')
    plt.show()
    

def plot_reconstruction(
    sample_idx,
    X_orig_var,
    X_recon_var,
    number_of_features,
    title_suffix='',
    start_time=None,
    end_time=None
):
    """
    Plots the original and reconstructed variable-length sequences for all features of a given sample.
    Shows 2 lines: original signal and VAE reconstruction.
    Displays features in a 2-column grid layout.
    """
    import seaborn as sns
    
    # Get original and reconstructed signals
    orig = X_orig_var[sample_idx]
    recon = X_recon_var[sample_idx]
    
    if orig.shape != recon.shape:
        print(f"Sample {sample_idx} shape mismatch: orig {orig.shape}, recon {recon.shape}")
        return
    
    # Apply time range slicing if specified
    if start_time is not None or end_time is not None:
        start_idx = start_time if start_time is not None else 0
        end_idx = end_time if end_time is not None else orig.shape[0]
        orig = orig[start_idx:end_idx]
        recon = recon[start_idx:end_idx]
        time_steps = np.arange(start_idx, start_idx + len(orig))
        time_suffix = f" (timesteps {start_idx}-{start_idx + len(orig)-1})"
    else:
        time_steps = np.arange(len(orig))
        time_suffix = ""
    
    # Compute MSE between original and reconstructed signal
    mse = np.mean((orig - recon) ** 2)
    
    print(f"Sample {sample_idx} Reconstruction MSE: {mse:.6f}{time_suffix}")

    # Define color palette
    colors = ['#2E86AB', '#F18F01']  # Blue, Orange
    
    # Calculate grid layout adaptively: use 1 column for single feature, 2 for multiple
    if number_of_features == 1:
        ncols = 1
        nrows = 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4))
        axes_flat = [axes]
    else:
        ncols = 2
        nrows = (number_of_features + ncols - 1) // ncols  # Ceiling division
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
        
        # Handle single row case
        if nrows == 1:
            axes = axes.reshape(1, -1)
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
    
    # Plot on all subplots
    for feature_idx in range(number_of_features):
        ax = axes_flat[feature_idx]
        
        # Extract feature signals
        orig_feat = orig[:, feature_idx]
        recon_feat = recon[:, feature_idx]
        
        # Plot both signals (only add labels on first subplot for single legend)
        if feature_idx == 0:
            ax.plot(time_steps, orig_feat, label='Original', color=colors[0], alpha=0.8, linewidth=1.5)
            ax.plot(time_steps, recon_feat, label='Reconstructed', color=colors[1], alpha=0.9, linewidth=2.0)
        else:
            ax.plot(time_steps, orig_feat, color=colors[0], alpha=0.8, linewidth=1.5)
            ax.plot(time_steps, recon_feat, color=colors[1], alpha=0.9, linewidth=2.0)
        
        ax.set_title(f'Feature {feature_idx} {title_suffix}', fontsize=12, pad=10)
        ax.set_xlabel('Time Steps', fontsize=10)
        ax.set_ylabel('Signal Amplitude', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Hide unused subplots
    for idx in range(number_of_features, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    # Add a single legend centered below all subplots
    fig.legend(['Original', 'Reconstructed'], loc='lower center', ncol=2, 
               frameon=True, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    
    plt.suptitle(f'Sample {sample_idx} Reconstruction{time_suffix} {title_suffix}', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.show()


def plot_reconstructions(X_orig, X_recon, sample=0, title_suffix=''):
    """
    Wrapper function to plot reconstructions. Handles both padded arrays and lists.
    """
    # Handle NaN padding
    if isinstance(X_orig, np.ndarray) and X_orig.dtype == object:
        orig_var = [remove_nan_padding(seq) for seq in X_orig]
        recon_var = X_recon
    elif isinstance(X_orig, np.ndarray):
        orig_var = [remove_nan_padding(X_orig[i]) for i in range(len(X_orig))]
        recon_var = X_recon
    else:
        orig_var = X_orig
        recon_var = X_recon
    
    # Get number of features
    num_features = orig_var[sample].shape[1]
    
    plot_reconstruction(sample, orig_var, recon_var, num_features, title_suffix=title_suffix)

def extract_variable_length_sequences(padded_data):
    """Extract valid sequences from NaN-padded sequences"""
    sequences = []
    for i in range(padded_data.shape[0]):
        seq = padded_data[i]
        valid_mask = ~np.isnan(seq).any(axis=1)
        sequences.append(seq[valid_mask])
    return sequences

def remove_nan_padding(sequence):
    """Remove rows where any feature is NaN."""
    return sequence[~np.isnan(sequence).any(axis=1)]

def fft_and_cull(signal, top_bins):
    """RFFT conversion and cull to lowest frequency bins (not by amplitude)."""
    fft_vals = np.fft.rfft(signal)
    fft_freqs = np.fft.rfftfreq(len(signal))
    fft_vals_culled = np.zeros_like(fft_vals)
    # Get indices of the lowest frequency bins (closest to zero)
    sorted_indices = np.argsort(np.abs(fft_freqs))
    low_freq_indices = sorted_indices[:top_bins]
    fft_vals_culled[low_freq_indices] = fft_vals[low_freq_indices]
    return fft_freqs, fft_vals, fft_vals_culled

def ifft_conversion(fft_vals_culled):
    """IRFFT conversion from culled RFFT values."""
    return np.fft.irfft(fft_vals_culled)

def plot_original_vs_reconstruction(signal, recon_signal, feature_idx, top_bins):
    """Plot original and reconstructed signals side by side with same scale."""
    min_y = min(np.min(signal), np.min(recon_signal))
    max_y = max(np.max(signal), np.max(recon_signal))
    fig, axs = plt.subplots(1, 2, figsize=(16, 4))
    axs[0].plot(signal, label="Original")
    axs[0].set_title(f"Original (Feature {feature_idx})")
    axs[0].set_xlabel("Time step")
    axs[0].set_ylabel("Value")
    axs[0].set_ylim(min_y, max_y)
    axs[0].legend()
    axs[1].plot(recon_signal, label=f"IRFFT Reconstruction (top {top_bins} bins)", color='orange')
    axs[1].set_title(f"IRFFT Reconstruction (Feature {feature_idx})")
    axs[1].set_xlabel("Time step")
    axs[1].set_ylabel("Value")
    axs[1].set_ylim(min_y, max_y)
    axs[1].legend()
    plt.tight_layout()
    plt.show()

def plot_culled_fft_bins(fft_freqs, fft_vals_culled, feature_idx, top_bins):
    import matplotlib.pyplot as plt
    amplitude = np.abs(fft_vals_culled)
    fig, axs = plt.subplots(1, 2, figsize=(16, 4))
    # Line plot, normal amplitude
    axs[0].plot(fft_freqs, amplitude, color='tab:blue')
    axs[0].set_title(f'Culled RFFT bins (Feature {feature_idx}, top {top_bins})')
    axs[0].set_xlabel('Frequency')
    axs[0].set_ylabel('Amplitude')
    # Line plot, log amplitude
    axs[1].plot(fft_freqs, amplitude, color='tab:orange')
    axs[1].set_title(f'Culled RFFT bins (Feature {feature_idx}, top {top_bins}) - Log-Scaled Amplitude')
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('Amplitude (log scale)')
    axs[1].set_yscale('log')
    plt.tight_layout()
    plt.show()
    

def dwt_resample_reconstruct_plot(sequence, feature_idx=0, wavelet='db4', level=4, fixed_cA_len=128, start_time=None, end_time=None):
    """
    For a single sequence (2D: [timesteps, features]), 
    - extracts DWT approximation coefficients,
    - reconstructs from cA (no resampling),
    - resamples cA, reconstructs,
    - plots all three: original, DWT recon (no resample), DWT recon (resampled).

    """
    
    # Remove NaN padding if present
    seq = sequence[~np.isnan(sequence).any(axis=1)]
    signal = seq[:, feature_idx]
    orig_len = len(signal)

    # DWT
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    cA = coeffs[0]
    detail_lens = [len(cD) for cD in coeffs[1:]]

    # DWT reconstruction (no resampling)
    cDs_zeros = [np.zeros(l) for l in detail_lens]
    coeffs_recon_noresample = [cA] + cDs_zeros
    recon_noresample = pywt.waverec(coeffs_recon_noresample, wavelet)
    recon_noresample = recon_noresample[:orig_len] if recon_noresample.size > orig_len else np.pad(recon_noresample, (0, orig_len - recon_noresample.size))

    # Resample cA
    x = np.linspace(0, 1, len(cA))
    f = interp1d(x, cA, kind='linear', fill_value='extrapolate')
    cA_resampled = f(np.linspace(0, 1, fixed_cA_len))

    # Unresample cA back to original length
    f_inv = interp1d(np.linspace(0, 1, fixed_cA_len), cA_resampled, kind='linear', fill_value='extrapolate')
    cA_orig = f_inv(np.linspace(0, 1, len(cA)))

    # DWT reconstruction (with resampling)
    coeffs_recon_resample = [cA_orig] + cDs_zeros
    recon_resample = pywt.waverec(coeffs_recon_resample, wavelet)
    recon_resample = recon_resample[:orig_len] if recon_resample.size > orig_len else np.pad(recon_resample, (0, orig_len - recon_resample.size))

    # Determine plot range
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = orig_len
    
    # Ensure valid range
    start_time = max(0, min(start_time, orig_len - 1))
    end_time = max(start_time + 1, min(end_time, orig_len))
    
    # Slice data for plotting
    signal_plot = signal[start_time:end_time]
    recon_noresample_plot = recon_noresample[start_time:end_time]
    recon_resample_plot = recon_resample[start_time:end_time]
    time_steps = np.arange(start_time, end_time)

    # Plot all three
    min_y = min(np.min(signal_plot), np.min(recon_noresample_plot), np.min(recon_resample_plot))
    max_y = max(np.max(signal_plot), np.max(recon_noresample_plot), np.max(recon_resample_plot))
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    axs[0].plot(time_steps, signal_plot, label="Original")
    axs[0].set_title(f"Original (Feature {feature_idx})")
    axs[0].set_ylim(min_y, max_y)
    axs[0].legend()
    axs[1].plot(time_steps, recon_noresample_plot, label="DWT Recon (no resample)", color='green')
    axs[1].set_title("DWT Recon (no resample)")
    axs[1].set_ylim(min_y, max_y)
    axs[1].legend()
    axs[2].plot(time_steps, recon_resample_plot, label=f"DWT Recon (resampled {fixed_cA_len})", color='orange')
    axs[2].set_title(f"DWT Recon (resampled {fixed_cA_len})")
    axs[2].set_ylim(min_y, max_y)
    axs[2].legend()
    for ax in axs:
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
    plt.tight_layout()
    plt.show()
    
def open_data_old(direc, ratio_train=0.8, dataset="ECG5000"):
    """Input:
    direc: location of the UCR archive
    ratio_train: ratio to split training and testset
    dataset: name of the dataset in the UCR archive"""
    datadir = direc + '/' + dataset + '/' + dataset
    data_train = np.loadtxt(datadir + '_TRAIN', delimiter=',')
    data_test_val = np.loadtxt(datadir + '_TEST', delimiter=',')[:-1]
    data = np.concatenate((data_train, data_test_val), axis=0)
    data = np.expand_dims(data, -1)

    N, D, _ = data.shape

    ind_cut = int(ratio_train * N)
    ind = np.random.permutation(N)
    return data[ind[:ind_cut], 1:, :], data[ind[ind_cut:], 1:, :], data[ind[:ind_cut], 0, :], data[ind[ind_cut:], 0, :]


def open_data(task="tlx", property="mean", use_nslr=False,
              ratio_train=0.8, max_seq_len=None, seed=42, pad_sequences=True, 
              keep_nan_padding=False, extractor='gaze', objective='cls'):
    """
    Load raw time-series from Tasks for VRAE classification.
    Combines gaze + pupil features, optionally pads sequences, splits train/val,
    and converts to time-series dataset format.
    pad_sequences: If False, returns lists of arrays. Default is True.
    keep_nan_padding: If True, keeps NaN values for padding instead of converting to 0.
    """

    # Load all tasks
    tasks = get_tasks(verbose=False)

    # Choose extractor
    if extractor == 'gaze':
        extractor = TemporalGazeFeatureExtractor()
    elif extractor == 'pupil':
        extractor = TemporalPupilFeatureExtractor()
        
    if objective == 'cls':
        labeller = TLXLabeller(objective="cls", property=property)
    else:
        labeller = TLXLabeller(objective="reg", property=property)
        
    dataset = ExperimentDataset(tasks, extractor, labeller, use_nslr=use_nslr)
    
    data = list(dataset)[0]

    X = data.X
    y = data.y

    if pad_sequences:
        X = to_time_series_dataset([np.array(j).transpose() for j in X.tolist()])
        if not keep_nan_padding:
            X = np.nan_to_num(X, nan=0)
        # If keep_nan_padding=True, we preserve NaN values for proper padding
    else:
        X = np.array([np.array(j).transpose() for j in X.tolist()], dtype=object)
        if not keep_nan_padding:
            X = [np.nan_to_num(x, nan=0) for x in X]

    # Train/validation split (stratified for classification, non-stratified for regression)
    if objective == 'cls':
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=ratio_train, random_state=seed, stratify=y
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=ratio_train, random_state=seed
        )

    return X_train, X_val, y_train, y_val


def load_all_tlx_subscales(use_nslr=False, ratio_train=0.8, max_seq_len=None, 
                           seed=42, pad_sequences=True, keep_nan_padding=False, 
                           extractor='gaze'):
    """
    Load data for all TLX subscales (mental, physical, temporal, performance, effort, frustration, mean).
    
    Returns:
        X_train, X_val: Training and validation data
        y_subscales_train, y_subscales_val: Dictionaries with keys for each subscale
    """
    subscales = ['mental', 'physical', 'temporal', 'performance', 'effort', 'frustration', 'mean']
    
    # Load data for each subscale
    y_subscales_train = {}
    y_subscales_val = {}
    
    for subscale in subscales:
        X_train, X_val, y_train, y_val = open_data(
            task="tlx", 
            property=subscale, 
            use_nslr=use_nslr,
            ratio_train=ratio_train, 
            max_seq_len=max_seq_len, 
            seed=seed, 
            pad_sequences=pad_sequences,
            keep_nan_padding=keep_nan_padding, 
            extractor=extractor, 
            objective='reg'
        )
        
        y_subscales_train[subscale] = y_train
        y_subscales_val[subscale] = y_val
    
    # All subscales will have the same X_train and X_val since they're based on the same data
    return X_train, X_val, y_subscales_train, y_subscales_val


def open_eseed_data(max_participants=48, max_videos=10,
                    pad_sequences=True, keep_nan_padding=False,
                    extractor='pupil', verbose=False):
    """
    Load full ESEED timeseries dataset (no train/test split).
    Returns:
        X: if pad_sequences True -> numpy array (n_series, max_len, n_features)
           else -> np.array(object) list of (timesteps, n_features)
    """
    from inputs.eseed_task import get_eseed_tasks
    from features import TemporalGazeFeatureExtractor, TemporalPupilFeatureExtractor

    # Get tasks
    tasks = get_eseed_tasks(max_participants=max_participants,
                            max_videos=max_videos,
                            verbose=verbose)

    # Choose extractor
    if extractor == 'gaze':
        ext = TemporalGazeFeatureExtractor()
    else:
        ext = TemporalPupilFeatureExtractor()

    X_list = []
    for task in tasks:
        try:
            feats = ext(task)
            # stack left/right pupil into (timesteps, 2)
            seq = np.stack([feats['lpd'], feats['rpd']], axis=1)
            X_list.append(seq)
        except Exception as e:
            if verbose:
                print(f"Skipping task {getattr(task, 'participant_id', '?')}/{getattr(task, 'video_id', '?')}: {e}")
            continue

    if pad_sequences:
        # Find max sequence length
        maxlen = max(seq.shape[0] for seq in X_list)
        n_features = X_list[0].shape[1]
        # Pad with NaN (or 0 if keep_nan_padding is False)
        pad_value = np.nan if keep_nan_padding else 0.0
        X_ts = np.full((len(X_list), maxlen, n_features), pad_value, dtype=float)
        for i, seq in enumerate(X_list):
            X_ts[i, :seq.shape[0], :] = seq
        return X_ts
    else:
        X_obj = np.array(X_list, dtype=object)
        if not keep_nan_padding:
            X_obj = [np.nan_to_num(x, nan=0.0) for x in X_obj]
        return X_obj

def pad_to_length(X, target_len):
    n_samples, seq_len, n_features = X.shape
    if seq_len == target_len:
        return X
    X_padded = np.full((n_samples, target_len, n_features), np.nan, dtype=X.dtype)
    X_padded[:, :seq_len, :] = X
    return X_padded

def save_vae_model(vrae, dload, filename="vrae_model.pt"):
    """
    Saves the VRAE model state_dict to the specified directory (legacy support).
    Args:
        vrae: Trained VRAE model instance.
        dload: Directory path to save the model.
        filename: Name of the saved file (default: 'vrae_model.pt').
    """
    os.makedirs(dload, exist_ok=True)
    save_path = os.path.join(dload, filename)
    torch.save(vrae.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def save_vae_checkpoint(trainer, vae_params, dload, filename="vae_checkpoint.pt"):
    """
    Saves a full VAE checkpoint with model weights and all metadata (Option 1: Research-style).
    Args:
        trainer: Trained VAE trainer instance (DWTVAETrainer, DWTVAERTrainer, etc.).
        vae_params: Dictionary of training parameters.
        dload: Directory path to save the checkpoint.
        filename: Name of the saved file (default: 'vae_checkpoint.pt').
    """
    os.makedirs(dload, exist_ok=True)
    save_path = os.path.join(dload, filename)
    
    checkpoint = {
        "model_state": trainer.model.state_dict(),
        "vae_params": vae_params,
        "num_features": trainer.num_features,
        "all_metas": trainer.all_metas,
        "wavelet": trainer.wavelet,
        "level": trainer.level,
        "fixed_cA_len": trainer.fixed_cA_len,
        "latent_dim": trainer.latent_dim,
        "normalization": "per-sequence z-score",
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def load_vae_checkpoint(checkpoint_path, Trainer):
    """
    Loads a VAE checkpoint and reconstructs the trainer and model.
    Args:
        checkpoint_path: Path to the checkpoint file.
        Trainer: Trainer class (DWTVAETrainer, DWTVAERTrainer, etc.).
    Returns:
        trainer: Reconstructed trainer instance with loaded weights and metadata.
    """
    ckpt = torch.load(checkpoint_path, weights_only=False)
    
    # Detect architecture from model weights
    # Conv: encoder.0.weight has 3 dims [out_channels, in_channels, kernel_size]
    # MLP: encoder.0.weight has 2 dims [out_features, in_features]
    # LSTM: encoder_lstm.weight_ih_l0 exists
    
    architecture = None
    if "encoder_lstm.weight_ih_l0" in ckpt["model_state"]:
        architecture = "lstm"
        # For LSTM, num_features is the input_size of the LSTM
        actual_num_features = ckpt["model_state"]["encoder_lstm.weight_ih_l0"].shape[1]
    elif len(ckpt["model_state"]["encoder.0.weight"].shape) == 3:
        architecture = "conv"
        # For Conv: encoder.0.weight has shape [out_channels, in_channels, kernel_size]
        actual_num_features = ckpt["model_state"]["encoder.0.weight"].shape[1]
    else:
        architecture = "mlp"
        # For MLP, we can't easily infer num_features from weights, use stored value
        actual_num_features = ckpt["num_features"]
    
    print(f"Detected architecture: {architecture.upper()}")
    
    # Check if stored num_features matches actual num_features
    if ckpt["num_features"] != actual_num_features:
        print(f"Warning: Stored num_features ({ckpt['num_features']}) doesn't match actual model "
              f"num_features ({actual_num_features}). Using actual value from model weights.")
        ckpt["num_features"] = actual_num_features
    
    # Backward compatibility: convert old 'use_conv' parameter to 'architecture'
    vae_params = ckpt["vae_params"].copy()
    if "use_conv" in vae_params:
        # Old checkpoint format with use_conv boolean
        use_conv = vae_params.pop("use_conv")
        if "architecture" not in vae_params:
            vae_params["architecture"] = "conv" if use_conv else "mlp"
        arch = vae_params["architecture"]
        print(f"Converted old 'use_conv={use_conv}' to 'architecture={arch}'")
    
    # Ensure architecture parameter matches detected architecture
    if "architecture" not in vae_params:
        vae_params["architecture"] = architecture
    
    # Instantiate trainer with saved params
    trainer = Trainer(**vae_params)
    
    # Restore metadata BEFORE creating the model
    trainer.num_features = ckpt["num_features"]
    trainer.all_metas = ckpt["all_metas"]
    
    # Import the correct model classes from the trainer's module
    trainer_module = Trainer.__module__
    
    if 'avaer' in trainer_module:
        from models.dwt_avaer import DWTMLPVAE, DWTConvVAE, DWTLSTMVAE
    elif 'avae' in trainer_module:
        from models.dwt_avae import DWTMLPVAE, DWTConvVAE, DWTLSTMVAE
    elif 'vaer' in trainer_module:
        from models.dwt_vaer import DWTMLPVAE, DWTConvVAE, DWTLSTMVAE
    elif 'vae' in trainer_module:
        from models.dwt_vae import DWTMLPVAE, DWTConvVAE, DWTLSTMVAE
    else:
        raise ValueError(f"Unknown trainer module: {trainer_module}")
    
    # Initialize model architecture based on detected architecture
    if architecture == "mlp":
        input_dim = ckpt["num_features"] * ckpt["fixed_cA_len"]
        hidden_dims = ckpt["vae_params"].get("hidden_dims", [1024, 512, 256])
        trainer.model = DWTMLPVAE(
            input_dim,
            ckpt["latent_dim"],
            hidden_dims
        ).to(trainer.device)
    elif architecture == "conv":
        trainer.model = DWTConvVAE(
            ckpt["num_features"],
            ckpt["fixed_cA_len"],
            ckpt["latent_dim"]
        ).to(trainer.device)
    elif architecture == "lstm":
        trainer.model = DWTLSTMVAE(
            ckpt["num_features"],
            ckpt["fixed_cA_len"],
            ckpt["latent_dim"]
        ).to(trainer.device)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Load model weights
    trainer.model.load_state_dict(ckpt["model_state"])
    
    return trainer

def compute_dwt_processed_signals(X_orig_var, wavelet='db4', level=4, fixed_cA_len=256):
    """
    Compute DWT-processed versions of signals (approximation-only reconstruction).
    This matches what the VAE sees after DWT processing.
    
    Args:
        X_orig_var (list of np.ndarray): Original sequences.
        wavelet (str): Wavelet type.
        level (int): Decomposition level.
        fixed_cA_len (int): Fixed length for resampling.
    Returns:
        list of np.ndarray: DWT-processed sequences.
    """
    dwt_processed_signals = []
    for orig in X_orig_var:
        orig_len, num_features = orig.shape
        processed_features = []
        
        for feature_idx in range(num_features):
            signal = orig[:, feature_idx]
            
            # DWT decomposition
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            cA = coeffs[0]
            detail_lens = [len(cD) for cD in coeffs[1:]]
            
            # Resample cA to fixed length and back
            x = np.linspace(0, 1, len(cA))
            f = interp1d(x, cA, kind='linear', fill_value='extrapolate')
            cA_resampled = f(np.linspace(0, 1, fixed_cA_len))
            
            # Unresample back
            f_inv = interp1d(np.linspace(0, 1, fixed_cA_len), cA_resampled, 
                           kind='linear', fill_value='extrapolate')
            cA_orig = f_inv(np.linspace(0, 1, len(cA)))
            
            # Reconstruct with zeros for detail coefficients
            cDs_zeros = [np.zeros(l) for l in detail_lens]
            coeffs_recon = [cA_orig] + cDs_zeros
            dwt_processed = pywt.waverec(coeffs_recon, wavelet)
            
            # Match original length
            if dwt_processed.size > orig_len:
                dwt_processed = dwt_processed[:orig_len]
            else:
                dwt_processed = np.pad(dwt_processed, (0, orig_len - dwt_processed.size))
            
            processed_features.append(dwt_processed)
        
        dwt_processed_signals.append(np.vstack(processed_features).T)
    return dwt_processed_signals

def compute_mse_loss(X_orig_var, X_recon_var):
    """
    Computes the mean squared error (MSE) loss between lists of variable-length sequences.
    Args:
        X_orig_var (list of np.ndarray): Original sequences.
        X_recon_var (list of np.ndarray): Reconstructed sequences.
    Returns:
        float: Mean MSE loss across all sequences.
    """
    losses = []
    for orig, recon in zip(X_orig_var, X_recon_var):
        if orig.shape == recon.shape:
            loss = np.mean((orig - recon) ** 2)
            losses.append(loss)
        else:
            # If shapes mismatch, skip or handle accordingly
            continue
    return np.mean(losses) if losses else float('nan')

def compute_dwt_freq_mse_loss(X_orig_var, X_recon_var, fixed_cA_len=256, wavelet='db4', level=4):
    """
    Computes the mean squared error (MSE) loss in the DWT frequency domain.
    Converts both original and reconstructed sequences to DWT features and computes MSE.
    Uses per-sequence averaging (Method 1).
    
    Args:
        X_orig_var (list of np.ndarray): Original sequences.
        X_recon_var (list of np.ndarray): Reconstructed sequences.
        fixed_cA_len (int): Fixed length for DWT approximation coefficients.
        wavelet (str): Wavelet type.
        level (int): Decomposition level.
    Returns:
        float: Mean MSE loss in DWT feature space, averaged per sequence.
    """
    from models.dwt_avaer import sequences_to_dwt
    
    # Normalize sequences
    sequences_orig_norm = []
    sequences_recon_norm = []
    
    for orig, recon in zip(X_orig_var, X_recon_var):
        if orig.shape != recon.shape:
            continue
        orig_np = np.asarray(orig, dtype=np.float32)
        recon_np = np.asarray(recon, dtype=np.float32)
        
        mu = orig_np.mean(axis=0, keepdims=True)
        sigma = orig_np.std(axis=0, keepdims=True) + 1e-7
        sequences_orig_norm.append((orig_np - mu) / sigma)
        sequences_recon_norm.append((recon_np - mu) / sigma)
    
    if len(sequences_orig_norm) == 0:
        return float('nan')
    
    # Convert to DWT features
    dwt_orig, _, _ = sequences_to_dwt(sequences_orig_norm, fixed_cA_len, wavelet, level)
    dwt_recon, _, _ = sequences_to_dwt(sequences_recon_norm, fixed_cA_len, wavelet, level)
    
    # Compute MSE per sequence, then average
    mse_per_seq = []
    for i in range(dwt_orig.shape[0]):
        seq_mse = torch.mean((dwt_orig[i] - dwt_recon[i]) ** 2).item()
        mse_per_seq.append(seq_mse)
    
    return np.mean(mse_per_seq)

def compute_reconstruction_loss(X_padded, X_recon_list):
    """
    Compute reconstruction loss between padded data and reconstructed sequences.
    Uses per-sequence MSE averaging.
    
    Args:
        X_padded: np.ndarray, shape (n_samples, max_len, n_features) with NaN padding
        X_recon_list: list of np.ndarray, each with shape (actual_len, n_features)
    
    Returns:
        float: Mean squared error averaged per sequence
    """
    mse_per_seq = []
    
    for i in range(len(X_recon_list)):
        # Extract original sequence (remove NaN padding)
        orig_seq = X_padded[i]
        valid_mask = ~np.isnan(orig_seq).any(axis=1)
        orig_valid = orig_seq[valid_mask]
        
        # Get reconstructed sequence
        recon_seq = X_recon_list[i]
        
        # Ensure same shape
        if orig_valid.shape != recon_seq.shape:
            print(f"Warning: Shape mismatch at sample {i}: {orig_valid.shape} vs {recon_seq.shape}")
            continue
        
        # Compute MSE for this sequence
        seq_mse = np.mean((orig_valid - recon_seq) ** 2)
        mse_per_seq.append(seq_mse)
    
    # Return average of per-sequence MSEs
    return np.mean(mse_per_seq) if mse_per_seq else float('nan')


def plot_predicted_vs_actual_comparison(model_name, X_data, y_data, dload='./model_dir', 
                                        n_subjects=47, n_subjects_per_group=6):
    """
    Plot predicted vs actual TLX scores comparing LNSO and K-Fold cross-validation.
    """

    warnings.filterwarnings('ignore')
    
    # Helper function to determine trainer class
    def get_trainer_class(model_filename):
        if "avaer" in model_filename:
            from models.dwt_avaer import DWTAVAERTrainer
            return DWTAVAERTrainer
        elif "avae" in model_filename:
            from models.dwt_avae import DWTAVAETrainer
            return DWTAVAETrainer
        elif "vaer" in model_filename:
            from models.dwt_vaer import DWTVAERTrainer
            return DWTVAERTrainer
        else:
            from models.dwt_vae import DWTVAETrainer
            return DWTVAETrainer
    
    # 1. Load the model
    model_path = str(Path(dload) / model_name)
    Trainer = get_trainer_class(model_name)
    vae = load_vae_checkpoint(model_path, Trainer)
    
    # 2. Encode the data to get latent representations
    z_all = vae.encode(X_data)
    
    # 3. Setup both cross-validation strategies
    samples_per_subject = len(X_data) // n_subjects
    group_ids = np.repeat(np.arange(n_subjects) // n_subjects_per_group, samples_per_subject)
    
    # 4. LNSO Cross-Validation
    cv_lnso = LeaveOneGroupOut()
    ridge_model = Ridge()
    y_pred_lnso = np.zeros_like(y_data)
    
    for train_idx, test_idx in cv_lnso.split(z_all, y_data, groups=group_ids):
        X_train_fold, X_test_fold = z_all[train_idx], z_all[test_idx]
        y_train_fold = y_data[train_idx]
        
        ridge_model.fit(X_train_fold, y_train_fold)
        y_pred_lnso[test_idx] = ridge_model.predict(X_test_fold)
    
    # 5. K-Fold Cross-Validation (5-fold)
    cv_kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_kfold = np.zeros_like(y_data)
    
    for train_idx, test_idx in cv_kfold.split(z_all):
        X_train_fold, X_test_fold = z_all[train_idx], z_all[test_idx]
        y_train_fold = y_data[train_idx]
        
        ridge_model.fit(X_train_fold, y_train_fold)
        y_pred_kfold[test_idx] = ridge_model.predict(X_test_fold)
    
    # 6. Plot both side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for idx, (ax, y_pred, cv_name) in enumerate([(axes[0], y_pred_lnso, 'LNSO'),
                                                   (axes[1], y_pred_kfold, 'K-Fold')]):
        # Scatter plot
        ax.scatter(y_data, y_pred, alpha=0.6, edgecolors='k', linewidths=0.5)
        
        # Best fit line
        z = np.polyfit(y_data, y_pred, 1)
        p = np.poly1d(z)
        x_line = np.linspace(y_data.min(), y_data.max(), 100)
        ax.plot(x_line, p(x_line), "r-", alpha=0.8, linewidth=2, 
                label=f'Best fit')
        
        # Perfect prediction line (diagonal)
        min_val = min(y_data.min(), y_pred.min())
        max_val = max(y_data.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, 
                label='Perfect prediction')
        
        # Calculate R²
        r2 = r2_score(y_data, y_pred)
        
        ax.set_xlabel('Actual Mean TLX Score', fontsize=12)
        ax.set_ylabel('Predicted Mean TLX Score', fontsize=12)
        ax.set_title(f'{model_name.replace(".pt", "")}: Ridge, {cv_name}\n', 
                     fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
    plt.suptitle('Comparison of Actual vs Predicted', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    
