import os
import sys
import numpy as np
import pandas as pd
import torch

from utils import open_data, akima_interpolate_zeros, compute_reconstruction_loss, save_vae_checkpoint
from models.dwt_avaer import DWTAVAERTrainer


def run_architecture_ablation_experiments():
    """Run ablation experiments with different model architectures (MLP, Conv, LSTM) and annealing methods."""
    
    # 1. Load and preprocess data
    X_train, X_val, y_train, y_val = open_data(extractor='pupil', keep_nan_padding=True, objective='reg')
    num_classes = len(np.unique(y_train))
    base = np.min(y_train)
    if base != 0:
        y_train -= base
        y_val -= base
    
    # Interpolate blinks
    X_train_interpolated = akima_interpolate_zeros(X_train, pad_value=np.nan)
    X_val_interpolated = akima_interpolate_zeros(X_val, pad_value=np.nan)
    X_train_interpolated = X_train_interpolated.astype(np.float32)
    X_val_interpolated = X_val_interpolated.astype(np.float32)
    
    # 2. Define base parameters
    base_parameters = {
        "latent_dim": 64,
        "lr": 1e-3,
        "epochs": 1000,
        "batch_size": 16,
        "wavelet": "sym16",
        "level": 3,
        "mode": "dwt",
        "device": "cuda",
        "num_cycles": 3
    }
    
    # 3. Define configurations for ablation study
    architectures = ['lstm']
    annealing_methods = ['cyclic', 'monotonic', None]
    
    # 4. Create model directory
    dload = './model_dir/ablations'
    os.makedirs(dload, exist_ok=True)
    
    # 5. Run all experiments
    train_results = []
    val_results = []
    
    for architecture in architectures:
        for annealing in annealing_methods:
            anneal_str = "none" if annealing is None else annealing
            
            print(f"\nTraining: architecture={architecture.upper()}, annealing={anneal_str}")
            
            # Update parameters
            params = {**base_parameters, 'architecture': architecture, 'annealing': annealing}
            
            # Train model
            trainer = DWTAVAERTrainer(**params)
            hist = trainer.fit(X_train_interpolated, tlx_targets=y_train)
            
            # Reconstruct
            recon_train = trainer.reconstruct(X_train_interpolated)
            recon_val = trainer.reconstruct(X_val_interpolated)
            
            # Compute losses
            train_loss = compute_reconstruction_loss(X_train_interpolated, recon_train)
            val_loss = compute_reconstruction_loss(X_val_interpolated, recon_val)
            
            # Save model
            fname = f"dwt_avaer_{architecture}_{anneal_str}.pt"
            save_vae_checkpoint(trainer, params, dload, fname)
            
            # Store results
            train_results.append({
                "Architecture": architecture.upper(),
                "Annealing": anneal_str,
                "Train_MSE": train_loss,
                "Model_File": fname
            })
            
            val_results.append({
                "Architecture": architecture.upper(),
                "Annealing": anneal_str,
                "Val_MSE": val_loss,
                "Model_File": fname
            })
    
    # 6. Create summary DataFrames
    train_df = pd.DataFrame(train_results)
    val_df = pd.DataFrame(val_results)
    
    # Merge train and val results
    summary_df = pd.merge(
        train_df, 
        val_df, 
        on=['Architecture', 'Annealing', 'Model_File']
    )
    
    return summary_df


if __name__ == "__main__":
    summary = run_architecture_ablation_experiments()
    print("\n" + "="*80)
    print(" "*15 + "ARCHITECTURE ABLATION STUDY")
    print("="*80)
    print(summary.to_string(index=False))
    print("="*80)