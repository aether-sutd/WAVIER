import os, warnings, glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from datasets.tlx import TLXClassDataset
from models.dwt_vae import DWTVAETrainer, sequences_to_dwt
from models.dwt_vaer import DWTVAERTrainer
from models.dwt_avaer import DWTAVAERTrainer
from models.dwt_avae import DWTAVAETrainer
from utils import save_vae_checkpoint, compute_mse_loss, compute_dwt_freq_mse_loss, compute_dwt_processed_signals, extract_variable_length_sequences, plot_history, load_vae_checkpoint, plot_reconstruction

def compute_validation_losses(vae, model_name, X_data, y_data, vae_params):
    """Compute all relevant validation losses for a trained VAE model.
    
    Returns a dict with MSE, KL, Aux, and Reg losses where applicable (None if not).
    """
    device = vae_params.get("device", "cuda")
    
    # Normalize and convert to DWT
    sequences = extract_variable_length_sequences(X_data)
    sequences_norm = []
    for seq in sequences:
        seq_np = np.asarray(seq, dtype=np.float32)
        mu = seq_np.mean(axis=0, keepdims=True)
        sigma = seq_np.std(axis=0, keepdims=True) + 1e-7
        sequences_norm.append((seq_np - mu) / sigma)
    
    dwt_data, _, _ = sequences_to_dwt(sequences_norm, vae_params["fixed_cA_len"], 
                                       vae_params["wavelet"], vae_params["level"])
    dwt_data = dwt_data.to(device)
    
    # Prepare TLX data if needed
    tlx_normalized = None
    if model_name in ["VAER", "AVAER"] and y_data is not None:
        tlx_mean = y_data.mean()
        tlx_std = y_data.std() + 1e-7
        tlx_normalized = (y_data - tlx_mean) / tlx_std
        tlx_tensor = torch.FloatTensor(tlx_normalized).to(device)
    
    vae.model.eval()
    losses = {"MSE": None, "KL": None, "Aux": None, "Reg": None}
    
    with torch.no_grad():
        # Forward pass based on model type
        if model_name in ["VAE", "AVAE"]:
            recon_dwt, mu, logvar = vae.model(dwt_data)
        else:  # VAER, AVAER
            recon_dwt, mu, logvar, y_mu, y_logvar = vae.model(dwt_data)
        
        # Reconstruction MSE
        recon_loss = F.mse_loss(recon_dwt, dwt_data, reduction='mean')
        losses["MSE"] = recon_loss.item()
        
        # KL divergence
        if model_name in ["VAER", "AVAER"] and tlx_normalized is not None:
            kl_loss = vae.model.conditional_kl_divergence(mu, logvar, tlx_tensor)
        else:
            kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1))
        losses["KL"] = kl_loss.item()
        
        # Auxiliary loss (AVAE models only)
        if model_name in ["AVAE", "AVAER"]:
            # Generate auxiliary samples (first forward pass for reconstruction)
            if model_name == "AVAER":
                aux_dwt, _, _, _, _ = vae.model(dwt_data)
                mu_aux, logvar_aux, _ = vae.model.encode(aux_dwt)
            else:  # AVAE
                aux_dwt, _, _ = vae.model(dwt_data)
                mu_aux, logvar_aux = vae.model.encode(aux_dwt)
            
            var = torch.exp(logvar)
            var_aux = torch.exp(logvar_aux)
            
            rho = vae.rho
            numerator = (var_aux + (rho ** 2) * var + torch.square(mu_aux - rho * mu))
            quadratic_term = torch.sum(numerator, dim=1) / (2.0 * (1.0 - rho ** 2))
            latent_dim = mu.shape[1]
            log_normalizer = -0.5 * latent_dim * np.log(2 * np.pi * (1.0 - rho ** 2))
            expected_log_conditional = torch.mean(log_normalizer - quadratic_term)
            entropy_aux = torch.mean(torch.sum(0.5 * torch.log(2 * np.pi * np.e * var_aux), dim=1))
            avae_term = expected_log_conditional + entropy_aux
            losses["Aux"] = avae_term.item()
        
        # Regression loss (VAER, AVAER models only)
        if model_name in ["VAER", "AVAER"] and tlx_normalized is not None:
            y_logvar_clamped = torch.clamp(y_logvar, min=-5, max=5)
            mse = (y_mu.squeeze() - tlx_tensor) ** 2
            precision = torch.exp(-y_logvar_clamped)
            reg_loss = 0.5 * torch.mean(y_logvar_clamped + mse * precision)
            losses["Reg"] = reg_loss.item()
    
    return losses

def run_all_vae_experiments(X_train, X_val, y_train, y_val, parameters, dload):
    configs = [
        ("VAE", DWTVAETrainer, None),
        ("VAE", DWTVAETrainer, "monotonic"),
        ("VAE", DWTVAETrainer, "cyclic"),
        ("VAER", DWTVAERTrainer, None),
        ("VAER", DWTVAERTrainer, "monotonic"),
        ("VAER", DWTVAERTrainer, "cyclic"),
        ("AVAER", DWTAVAERTrainer, None),
        ("AVAER", DWTAVAERTrainer, "monotonic"),
        ("AVAER", DWTAVAERTrainer, "cyclic"),
        ("AVAE", DWTAVAETrainer, None),
        ("AVAE", DWTAVAETrainer, "monotonic"),
        ("AVAE", DWTAVAETrainer, "cyclic"),
    ]
    train_results = []
    val_results = []
    print("\n" + "="*80)
    print(" "*20 + "VAE TRAINING EXPERIMENTS")
    print("="*80)
    
    for idx, (model_name, Trainer, annealing) in enumerate(configs, 1):
        anneal_str = "none" if annealing is None else annealing
        print(f"\n{'─'*80}")
        print(f"[{idx}/12] Training {model_name} with {anneal_str} annealing")
        print(f"{'─'*80}")
        
        # Store parameters for checkpoint saving
        vae_params = {
            "fixed_cA_len": parameters.get("fixed_cA_len", 256),
            "latent_dim": parameters.get("latent_dim", 64),
            "lr": parameters.get("lr", 1e-3),
            "epochs": parameters.get("epochs", 1000),
            "batch_size": parameters.get("batch_size", 16),
            "wavelet": parameters.get("wavelet", 'db4'),
            "level": parameters.get("dwt_level", 4),
            "annealing": annealing,
            "use_conv": parameters.get("use_conv", True),
            "device": parameters.get("device", "cuda"),
            "num_cycles": parameters.get("num_cycles", 10)
        }
        
        vae = Trainer(**vae_params)
        # VAE and AVAE don't take tlx_targets, only VAER and AVAER do
        if model_name in ["VAE", "AVAE"]:
            hist = vae.fit(X_train)
        else:
            hist = vae.fit(X_train, tlx_targets=y_train)
        
        # Plot training history
        plot_history(hist['total_loss'])
        
        # Compute train losses
        train_losses = compute_validation_losses(vae, model_name, X_train, y_train, vae_params)
        
        # Compute val losses  
        val_losses = compute_validation_losses(vae, model_name, X_val, y_val, vae_params)
        
        # Also compute reconstruction MSE on actual time-domain signals
        X_train_recon = vae.reconstruct(X_train)
        X_val_recon = vae.reconstruct(X_val)
        X_train_orig_var = extract_variable_length_sequences(X_train)
        X_val_orig_var = extract_variable_length_sequences(X_val)
        
        X_train_dwt_processed = compute_dwt_processed_signals(
            X_train_orig_var, vae_params["wavelet"], vae_params["level"], vae_params["fixed_cA_len"]
        )
        X_val_dwt_processed = compute_dwt_processed_signals(
            X_val_orig_var, vae_params["wavelet"], vae_params["level"], vae_params["fixed_cA_len"]
        )
        
        train_time_mse = compute_mse_loss(X_train_dwt_processed, X_train_recon)
        val_time_mse = compute_mse_loss(X_val_dwt_processed, X_val_recon)
        
        fname = f"dwt_{model_name.lower()}_{anneal_str}.pt"
        save_vae_checkpoint(vae, vae_params, dload, fname)
        print(f"✓ Train MSE: {train_time_mse:.6f} | Val MSE: {val_time_mse:.6f}\n")
        
        # Store results
        train_results.append({
            "Model": model_name,
            "Annealing": anneal_str,
            "MSE": train_time_mse,
            "KL": train_losses["KL"],
            "Aux": train_losses["Aux"] if train_losses["Aux"] is not None else np.nan,
            "Reg": train_losses["Reg"] if train_losses["Reg"] is not None else np.nan,
            "Model_File": fname
        })
        
        val_results.append({
            "Model": model_name,
            "Annealing": anneal_str,
            "MSE": val_time_mse,
            "KL": val_losses["KL"],
            "Aux": val_losses["Aux"] if val_losses["Aux"] is not None else np.nan,
            "Reg": val_losses["Reg"] if val_losses["Reg"] is not None else np.nan,
            "Model_File": fname
        })
    
    return pd.DataFrame(train_results), pd.DataFrame(val_results)

def load_and_visualize_model(model_filename, X_data, dload='./model_dir', sample_idx=3, 
                            start_time=None, end_time=None):
    """
    Load a VAE checkpoint and visualize reconstruction for a sample.
    
    Args:
        model_filename: Checkpoint filename (e.g., "dwt_vae_none.pt", "dwt_avaer_cyclic.pt")
        X_data: Data to reconstruct (e.g., X_val_interpolated)
        dload: Directory containing checkpoints (default: './model_dir')
        sample_idx: Sample index to visualize (default: 3)
        start_time: Start timestep for visualization (default: None = from beginning)
        end_time: End timestep for visualization (default: None = to end)
    
    Returns:
        vae: Loaded VAE trainer instance
    """
    # Map model name to appropriate Trainer class
    # Order matters! Check most specific first
    trainer_map = [
        ("avaer", DWTAVAERTrainer),
        ("avae", DWTAVAETrainer),
        ("vaer", DWTVAERTrainer),
        ("vae", DWTVAETrainer),
    ]
    
    # Determine model type from filename
    model_type = None
    Trainer = None
    for key, trainer_class in trainer_map:
        if key in model_filename.lower():
            model_type = key
            Trainer = trainer_class
            break
    
    if Trainer is None:
        raise ValueError(f"Could not determine model type from filename: {model_filename}")
    
    # Load checkpoint
    checkpoint_path = os.path.join(dload, model_filename)
    vae = load_vae_checkpoint(checkpoint_path, Trainer)
    
    # Reconstruct data
    X_recon = vae.reconstruct(X_data)
    X_orig_var = extract_variable_length_sequences(X_data)
    
    # Handle both object arrays (HTC) and normal 3D arrays (COLET)
    if X_data.dtype == object:
        # Object array: each element is (timesteps, features)
        number_of_features = X_data[0].shape[1]
    else:
        # Normal 3D array: (n_samples, timesteps, features)
        number_of_features = X_data.shape[2]
    
    # Plot reconstruction
    plot_reconstruction(
        sample_idx=sample_idx,
        X_orig_var=X_orig_var,
        X_recon_var=X_recon,
        number_of_features=number_of_features,
        start_time=start_time,
        end_time=end_time
    )
    

def plot_tsne_latent_space(model_filename, X_data, labels, dload='./model_dir', 
                           perplexity=15, figsize=(8, 6), show_r2=True):
    """Plot t-SNE visualization of VAE latent space with optional R² score."""
    trainer_map = [("avaer", DWTAVAERTrainer), ("avae", DWTAVAETrainer),
                   ("vaer", DWTVAERTrainer), ("vae", DWTVAETrainer)]
    
    Trainer = None
    for key, trainer_class in trainer_map:
        if key in model_filename.lower():
            Trainer = trainer_class
            break
    
    if Trainer is None:
        raise ValueError(f"Could not determine model type from filename: {model_filename}")
    
    # Load and encode
    vae = load_vae_checkpoint(os.path.join(dload, model_filename), Trainer)
    z_all = vae.encode(X_data)
    
    # Calculate R²
    reg = LinearRegression()
    reg.fit(z_all, labels.reshape(-1, 1))
    r2 = r2_score(labels, reg.predict(z_all))
    
    # t-SNE
    z_tsne = TSNE(perplexity=perplexity, min_grad_norm=1E-12, 
                  max_iter=3000, random_state=42).fit_transform(z_all)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(z_tsne[:, 0], z_tsne[:, 1], c=labels, 
                        cmap='RdBu', marker='*', s=50, alpha=0.7, linewidths=0)
    
    title = f"{model_filename.replace('.pt', '').replace('dwt_', '').upper()}"
    if show_r2:
        title += f"\nR² = {r2:.4f}"
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=10)
    ax.set_ylabel('t-SNE 2', fontsize=10)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('TLX Score', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def latent_regression_lnso(X_all, y_all_subscales, dload='./model_dir', n_subjects=47, n_subjects_per_group=6, metric='mse', save_csv=None):
    """Evaluate VAE latent representations using LNSO cross-validation with 10 regression models across all TLX subscales.
    
    Args:
        X_all: All data
        y_all_subscales: Dict with keys 'mental', 'physical', 'temporal', 'performance', 'effort', 'frustration', 'mean'
        dload: Directory containing VAE checkpoints
        n_subjects: Total number of subjects
        n_subjects_per_group: Number of subjects per group for LNSO
        metric: Evaluation metric - 'mse' for Mean Squared Error or 'r2' for R² score (default: 'mse')
        save_csv: Path to save results CSV (optional)
    
    Returns:
        DataFrame with metric scores for each VAE model, subscale, regression models, best model and best metric score.
    """
    warnings.filterwarnings('ignore', category=UserWarning)
    
    def get_trainer_class(model_name):
        if "avaer" in model_name:
            return DWTAVAERTrainer
        elif "avae" in model_name:
            return DWTAVAETrainer
        elif "vaer" in model_name:
            return DWTVAERTrainer
        return DWTVAETrainer
    
    model_files = sorted(Path(dload).glob("*.pt"))
    samples_per_subject = np.full(n_subjects, len(X_all) // n_subjects)
    
    group_ids = np.repeat(np.arange(n_subjects) // n_subjects_per_group, samples_per_subject[0])
    
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "ElasticNet": ElasticNet(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "HistGB": HistGradientBoostingRegressor(random_state=42),
        "SVR": SVR(),
        "XGB": XGBRegressor(use_label_encoder=False, verbosity=0, random_state=42),
        "LGBM": LGBMRegressor(verbose=-1, random_state=42),
        "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
    }
    
    subscales = ['mental', 'physical', 'temporal', 'performance', 'effort', 'frustration', 'mean']
    all_results = []
    cv = LeaveOneGroupOut()
    
    # Set scoring method and metric labels based on metric parameter
    if metric == 'r2':
        scoring = 'r2'
        metric_label = 'R²'
        best_metric_init = float('-inf')  # Higher is better for R²
        is_higher_better = True
    else:  # metric == 'mse'
        scoring = 'neg_mean_squared_error'
        metric_label = 'MSE'
        best_metric_init = float('inf')  # Lower is better for MSE
        is_higher_better = False
    
    for model_path in model_files:
        vae = load_vae_checkpoint(str(model_path), get_trainer_class(model_path.name))
        z_all = vae.encode(X_all)
        
        for subscale in subscales:
            y_all = y_all_subscales[subscale]
            
            results = {}
            best_metric_value = best_metric_init
            best_model_name = None
            
            for model_name, model in models.items():
                scores = cross_val_score(model, z_all, y_all, groups=group_ids, cv=cv, scoring=scoring)
                mean_score = scores.mean() if metric == 'r2' else -scores.mean()
                results[model_name] = mean_score
                
                if (is_higher_better and mean_score > best_metric_value) or (not is_higher_better and mean_score < best_metric_value):
                    best_metric_value = mean_score
                    best_model_name = model_name
            
            results['Best Model'] = best_model_name
            results[f'Best {metric_label}'] = best_metric_value
            all_results.append({'Model': model_path.name, 'Subscale': subscale, **results})
    
    reg_models = ["Linear", "Ridge", "ElasticNet", "Random Forest", "Gradient Boosting", 
                  "HistGB", "SVR", "XGB", "LGBM", "CatBoost"]
    results_df = pd.DataFrame(all_results)
    results_df = results_df[['Model', 'Subscale'] + reg_models + ['Best Model', f'Best {metric_label}']]
    
    # Save to CSV if requested
    if save_csv is not None:
        results_df.to_csv(save_csv, index=False)
        print(f"Results saved to {save_csv}")
    
    return results_df


def latent_regression_kfold(X_all, y_all_subscales, dload='./model_dir', folds=5, random_state=42, metric='mse', save_csv=None):
    """Evaluate VAE latent representations using K-Fold cross-validation with 10 regression models across all TLX subscales.
    
    Args:
        X_all: All data
        y_all_subscales: Dict with keys 'mental', 'physical', 'temporal', 'performance', 'effort', 'frustration', 'mean'
        dload: Directory containing VAE checkpoints
        folds: Number of folds for cross-validation
        random_state: Random seed for reproducibility
        metric: Evaluation metric - 'mse' for Mean Squared Error or 'r2' for R² score (default: 'mse')
        save_csv: Path to save results CSV (optional)
    
    Returns:
        DataFrame with metric scores for each VAE model, subscale, regression models, best model and best metric score.
    """
    warnings.filterwarnings('ignore', category=UserWarning)
    
    def get_trainer_class(model_name):
        if "avaer" in model_name:
            return DWTAVAERTrainer
        elif "avae" in model_name:
            return DWTAVAETrainer
        elif "vaer" in model_name:
            return DWTVAERTrainer
        return DWTVAETrainer
    
    model_files = sorted(Path(dload).glob("*.pt"))
    
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "ElasticNet": ElasticNet(),
        "Random Forest": RandomForestRegressor(random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
        "HistGB": HistGradientBoostingRegressor(random_state=random_state),
        "SVR": SVR(),
        "XGB": XGBRegressor(use_label_encoder=False, verbosity=0, random_state=random_state),
        "LGBM": LGBMRegressor(verbose=-1, random_state=random_state),
        "CatBoost": CatBoostRegressor(verbose=0, random_state=random_state)
    }
    
    subscales = ['mental', 'physical', 'temporal', 'performance', 'effort', 'frustration', 'mean']
    all_results = []
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    
    # Set scoring method and metric labels based on metric parameter
    if metric == 'r2':
        scoring = 'r2'
        metric_label = 'R²'
        best_metric_init = float('-inf')  # Higher is better for R²
        is_higher_better = True
    else:  # metric == 'mse'
        scoring = 'neg_mean_squared_error'
        metric_label = 'MSE'
        best_metric_init = float('inf')  # Lower is better for MSE
        is_higher_better = False
    
    for model_path in model_files:
        vae = load_vae_checkpoint(str(model_path), get_trainer_class(model_path.name))
        z_all = vae.encode(X_all)
        
        for subscale in subscales:
            y_all = y_all_subscales[subscale]
            
            results = {}
            best_metric_value = best_metric_init
            best_model_name = None
            
            for model_name, model in models.items():
                scores = cross_val_score(model, z_all, y_all, cv=kf, scoring=scoring)
                mean_score = scores.mean() if metric == 'r2' else -scores.mean()
                results[model_name] = mean_score
                
                if (is_higher_better and mean_score > best_metric_value) or (not is_higher_better and mean_score < best_metric_value):
                    best_metric_value = mean_score
                    best_model_name = model_name
            
            results['Best Model'] = best_model_name
            results[f'Best {metric_label}'] = best_metric_value
            all_results.append({'Model': model_path.name, 'Subscale': subscale, **results})
    
    reg_models = ["Linear", "Ridge", "ElasticNet", "Random Forest", "Gradient Boosting", 
                  "HistGB", "SVR", "XGB", "LGBM", "CatBoost"]
    results_df = pd.DataFrame(all_results)
    results_df = results_df[['Model', 'Subscale'] + reg_models + ['Best Model', f'Best {metric_label}']]
    
    # Save to CSV if requested
    if save_csv is not None:
        results_df.to_csv(save_csv, index=False)
        print(f"Results saved to {save_csv}")
    
    return results_df

def plot_all_models_latent_space(X_data, labels, dload='./model_dir', perplexity=15):
    """Plot t-SNE visualizations for all 12 VAE models in 4×3 grid."""
    warnings.filterwarnings('ignore')
    
    model_types = ['vae', 'vaer', 'avaer', 'avae']
    model_names = ['VAE', 'VAER', 'AVAER', 'AVAE']
    annealing_types = ['none', 'monotonic', 'cyclic']
    trainer_map = {'vae': DWTVAETrainer, 'vaer': DWTVAERTrainer,
                   'avaer': DWTAVAERTrainer, 'avae': DWTAVAETrainer}
    
    # Get actual min/max from labels for color scale
    vmin = np.min(labels)
    vmax = np.max(labels)
    
    # Create figure with proper gridspec for equal-sized subplots
    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(4, 4, figure=fig, wspace=0.25, hspace=0.35, 
                          width_ratios=[1, 1, 1, 0.05],
                          top=0.98, bottom=0.02, left=0.05, right=0.95)
    
    for i, (model_type, model_name) in enumerate(zip(model_types, model_names)):
        for j, annealing in enumerate(annealing_types):
            ax = fig.add_subplot(gs[i, j])
            model_filename = f"dwt_{model_type}_{annealing}.pt"
            
            try:
                vae = load_vae_checkpoint(os.path.join(dload, model_filename), 
                                         trainer_map[model_type])
                z_all = vae.encode(X_data)
                
                reg = LinearRegression()
                reg.fit(z_all, labels.reshape(-1, 1))
                r2 = r2_score(labels, reg.predict(z_all))
                
                z_tsne = TSNE(perplexity=perplexity, min_grad_norm=1E-12, 
                             max_iter=3000, random_state=42).fit_transform(z_all)
                
                scatter = ax.scatter(z_tsne[:, 0], z_tsne[:, 1], c=labels, 
                                   cmap='RdBu', marker='*', s=30, alpha=0.7, 
                                   linewidths=0, vmin=vmin, vmax=vmax)
                
                ax.set_title(f"{model_name} - {annealing.capitalize()}\nR² = {r2:.4f}", 
                           fontsize=11, fontweight='bold', pad=10)
                ax.set_xlabel('t-SNE 1', fontsize=10)
                ax.set_ylabel('t-SNE 2', fontsize=10)
                ax.tick_params(labelsize=9)
                ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
                
                # Add colorbar to rightmost column
                if j == 2:
                    cax = fig.add_subplot(gs[i, 3])
                    cbar = plt.colorbar(scatter, cax=cax)
                    cbar.set_label('TLX Score', fontsize=10)
                    cbar.ax.tick_params(labelsize=9)
                    
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading\n{model_filename}", 
                       ha='center', va='center', transform=ax.transAxes, 
                       fontsize=10, color='red')
                ax.set_title(f"{model_name} - {annealing.capitalize()}", 
                           fontsize=11, fontweight='bold', pad=10)
    
    plt.show()

def latent_classification_lsno(X_all, y_all, dload='./model_dir', n_subjects_per_group=6, save_csv=None):
    """Evaluate all VAE models across all classification configurations using LNSO cross-validation.

    """
    warnings.filterwarnings('ignore', category=UserWarning)
    
    def get_trainer_class(model_name):
        if "avaer" in model_name:
            return DWTAVAERTrainer
        elif "avae" in model_name:
            return DWTAVAETrainer
        elif "vaer" in model_name:
            return DWTVAERTrainer
        return DWTVAETrainer
    
    def get_classification_config(X, y, config: str):
        """Extract the classification configuration logic from TLXClassDataset"""
        samples_per_subj = np.full(47, 4)

        if config == "C1vC2":
            reshaped = (y != 2).reshape(-1, 4)
            samples_per_subj = np.sum(reshaped, axis=1)
            X = X[y != 2]
            y = y[y != 2]
            return X, y, samples_per_subj
        
        if config == "C1vC3":
            reshaped = (y != 1).reshape(-1, 4)
            samples_per_subj = np.sum(reshaped, axis=1)
            X = X[y != 1]
            y = (y[y != 1] > 0).astype(int)
            return X, y, samples_per_subj
        
        if config == "C2vC3":
            reshaped = (y != 0).reshape(-1, 4)
            samples_per_subj = np.sum(reshaped, axis=1)
            X = X[y != 0]
            y = y[y != 0] - 1
            return X, y, samples_per_subj
        
        if config == "all":
            return X, y, samples_per_subj
        
        if config == "C1C2vC3":
            y = (y == 2).astype(int)
            return X, y, samples_per_subj
        
        if config == "C1vC2C3":
            y = (y != 0).astype(int)
            return X, y, samples_per_subj

        return X, y, samples_per_subj
    
    # Get all model files
    model_files = sorted(glob.glob(os.path.join(dload, "*.pt")))
    
    # Classification models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "HistGB": HistGradientBoostingClassifier(),
        "SVM": SVC(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "XGB": XGBClassifier(use_label_encoder=False, verbosity=0),
        "LGBM": LGBMClassifier(verbose=-1),
        "CatBoost": CatBoostClassifier(verbose=0)
    }
    
    all_results = []
    cv = LeaveOneGroupOut()
    
    for model_path in model_files:
        model_name = os.path.basename(model_path).replace('.pt', '')
        
        # Load VAE once per model
        vae = load_vae_checkpoint(str(model_path), get_trainer_class(model_name))
        
        # Iterate over all classification configurations
        configs = ["C1vC2", "C1vC3", "C2vC3", "C1C2vC3", "C1vC2C3", "all"]
        for config_name in configs:
            # Get data split for this config
            X_split, y_split, samples_per_subj = get_classification_config(
                X_all.copy(), 
                y_all.copy(), 
                config_name
            )
            
            # Encode with VAE
            z_all = vae.encode(X_split)
            
            # Setup LNSO cross-validation
            n_subjects = len(samples_per_subj)
            group_ids = []
            for subj_idx in range(n_subjects):
                group_id = subj_idx // n_subjects_per_group
                group_ids.extend([group_id] * samples_per_subj[subj_idx])
            group_ids = np.array(group_ids)
            
            # Evaluate all classifiers
            result_dict = {'Model': model_name, 'Config': config_name}
            best_accuracy = -float('inf')
            best_model_name = None
            
            for clf_name, clf_model in models.items():
                scores = cross_val_score(clf_model, z_all, y_split, groups=group_ids, cv=cv, scoring='accuracy')
                mean_accuracy = scores.mean()
                result_dict[clf_name] = mean_accuracy
                
                if mean_accuracy > best_accuracy:
                    best_accuracy = mean_accuracy
                    best_model_name = clf_name
            
            result_dict['Best Model'] = best_model_name
            result_dict['Best Accuracy'] = best_accuracy
            
            all_results.append(result_dict)
    
    # Create DataFrame with proper column order
    cls_models = ["Logistic Regression", "Decision Tree", "Random Forest", 
                  "Gradient Boosting", "HistGB", "SVM", "K-Nearest Neighbors",
                  "Naive Bayes", "XGB", "LGBM", "CatBoost"]
    results_df = pd.DataFrame(all_results)
    results_df = results_df[['Model', 'Config'] + cls_models + ['Best Model', 'Best Accuracy']]
    
    # Save to CSV if requested
    if save_csv is not None:
        results_df.to_csv(save_csv, index=False)
        print(f"Results saved to: {save_csv}")
    
    return results_df

def latent_classification_kfold(X_all, y_all, dload='./model_dir', folds=5, random_state=42, save_csv=None):
    """Evaluate all VAE models across all classification configurations using K-Fold cross-validation.
    
    Args:
        X_all: All data
        y_all: All labels (classification labels)
        dload: Directory containing VAE checkpoints
        folds: Number of folds for cross-validation
        random_state: Random seed for reproducibility
        save_csv: Path to save results CSV (optional)
    
    Returns:
        DataFrame with accuracy scores for each VAE model, config, classifiers, best model and best accuracy.
    """
    warnings.filterwarnings('ignore', category=UserWarning)
    
    def get_trainer_class(model_name):
        if "avaer" in model_name:
            return DWTAVAERTrainer
        elif "avae" in model_name:
            return DWTAVAETrainer
        elif "vaer" in model_name:
            return DWTVAERTrainer
        return DWTVAETrainer
    
    def get_classification_config(X, y, config: str):
        """Extract the classification configuration logic from TLXClassDataset"""
        if config == "C1vC2":
            X = X[y != 2]
            y = y[y != 2]
            return X, y
        
        if config == "C1vC3":
            X = X[y != 1]
            y = (y[y != 1] > 0).astype(int)
            return X, y
        
        if config == "C2vC3":
            X = X[y != 0]
            y = y[y != 0] - 1
            return X, y
        
        if config == "all":
            return X, y
        
        if config == "C1C2vC3":
            y = (y == 2).astype(int)
            return X, y
        
        if config == "C1vC2C3":
            y = (y != 0).astype(int)
            return X, y

        return X, y
    
    # Get all model files
    model_files = sorted(glob.glob(os.path.join(dload, "*.pt")))
    
    # Classification models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "HistGB": HistGradientBoostingClassifier(random_state=random_state),
        "SVM": SVC(random_state=random_state),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "XGB": XGBClassifier(use_label_encoder=False, verbosity=0, random_state=random_state),
        "LGBM": LGBMClassifier(verbose=-1, random_state=random_state),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=random_state)
    }
    
    all_results = []
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    
    for model_path in model_files:
        model_name = os.path.basename(model_path).replace('.pt', '')
        
        # Load VAE once per model
        vae = load_vae_checkpoint(str(model_path), get_trainer_class(model_name))
        
        # Iterate over all classification configurations
        configs = ["C1vC2", "C1vC3", "C2vC3", "C1C2vC3", "C1vC2C3", "all"]
        for config_name in configs:
            # Get data split for this config
            X_split, y_split = get_classification_config(
                X_all.copy(), 
                y_all.copy(), 
                config_name
            )
            
            # Encode with VAE
            z_all = vae.encode(X_split)
            
            # Evaluate all classifiers
            result_dict = {'Model': model_name, 'Config': config_name}
            best_accuracy = -float('inf')
            best_model_name = None
            
            for clf_name, clf_model in models.items():
                scores = cross_val_score(clf_model, z_all, y_split, cv=kf, scoring='accuracy')
                mean_accuracy = scores.mean()
                result_dict[clf_name] = mean_accuracy
                
                if mean_accuracy > best_accuracy:
                    best_accuracy = mean_accuracy
                    best_model_name = clf_name
            
            result_dict['Best Model'] = best_model_name
            result_dict['Best Accuracy'] = best_accuracy
            
            all_results.append(result_dict)
    
    # Create DataFrame with proper column order
    cls_models = ["Logistic Regression", "Decision Tree", "Random Forest", 
                  "Gradient Boosting", "HistGB", "SVM", "K-Nearest Neighbors",
                  "Naive Bayes", "XGB", "LGBM", "CatBoost"]
    results_df = pd.DataFrame(all_results)
    results_df = results_df[['Model', 'Config'] + cls_models + ['Best Model', 'Best Accuracy']]
    
    # Save to CSV if requested
    if save_csv is not None:
        results_df.to_csv(save_csv, index=False)
        print(f"Results saved to: {save_csv}")
    
    return results_df
