import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pywt
from scipy.interpolate import interp1d
from utils import *

# -------------------------
# Helper functions
# -------------------------

def dwt_to_time_torch(recon_dwt, lengths, num_features, fixed_cA_len=256, device='cpu'):
    """Convert reconstructed DWT features back to variable-length sequences."""
    batch_size = recon_dwt.shape[0]
    recon_time = []
    all = recon_dwt.view(batch_size, num_features, fixed_cA_len)
    
    for i in range(batch_size):
        seq_len = int(lengths[i].item()) if isinstance(lengths[i], torch.Tensor) else int(lengths[i])
        # Placeholder implementation - returns zeros for now
        time_seq = torch.zeros(num_features, seq_len, device=device)
        recon_time.append(time_seq.permute(1, 0).contiguous())
    return recon_time

def remove_nan_padding(sample):
    """Remove trailing all-NaN rows from 2D array."""
    if sample.ndim != 2:
        raise ValueError("sample must be 2D")
    valid = ~np.all(np.isnan(sample), axis=1)
    if not np.any(valid):
        return np.zeros((0, sample.shape[1]), dtype=sample.dtype)
    last = np.where(valid)[0].max()
    return sample[: last + 1, :]

def resample_coeffs(cA, fixed_len):
    """Resample 1D coefficient array to fixed length using linear interpolation."""
    if len(cA) == fixed_len:
        return cA
    x = np.linspace(0, 1, len(cA))
    f = interp1d(x, cA, kind='linear', fill_value='extrapolate')
    return f(np.linspace(0, 1, fixed_len))

def sample_to_dwt_approx_only(sample, wavelet='db4', level=4):
    """Extract DWT approximation coefficients and metadata for each feature."""
    sample = np.asarray(sample)
    T, F = sample.shape
    approx_parts = []
    feature_meta = []
    
    for f in range(F):
        sig = sample[:, f]
        coeffs = pywt.wavedec(sig, wavelet, level=level)
        cA = coeffs[0]
        detail_lens = [len(cD) for cD in coeffs[1:]]
        approx_parts.append(cA)
        feature_meta.append({
            'orig_len': T,
            'cA_len': len(cA),
            'detail_lens': detail_lens,
            'level': level,
            'wavelet': wavelet
        })
    
    meta = {'num_features': F, 'feature_meta': feature_meta}
    return approx_parts, meta

def sequences_to_dwt(sequences, fixed_cA_len=256, wavelet='db4', level=4):
    """Convert variable-length sequences to fixed-length DWT feature vectors."""
    dwt_features = []
    all_metas = []
    if len(sequences) == 0:
        return torch.FloatTensor([]), [], 0, []
    
    num_features = sequences[0].shape[1]
    for seq in sequences:
        approx_parts, meta = sample_to_dwt_approx_only(seq, wavelet=wavelet, level=level)
        resampled_parts = [resample_coeffs(cA, fixed_cA_len) for cA in approx_parts]
        feat_vec = np.concatenate(resampled_parts)
        dwt_features.append(feat_vec)
        all_metas.append(meta)
    
    return torch.FloatTensor(np.array(dwt_features)), all_metas, num_features

def unresample_and_reconstruct_single(feat_vec, meta, fixed_cA_len):
    """Reconstruct time-domain signal from DWT feature vector."""
    F = meta['num_features']
    recon_features = []
    idx = 0

    for f in range(F):
        fm = meta['feature_meta'][f]
        cA_len = fm['cA_len']
        wavelet = fm['wavelet']
        detail_lens = fm['detail_lens']
        orig_len = fm['orig_len']

        # Extract and unresample coefficients
        cA_resampled = feat_vec[idx: idx + fixed_cA_len]
        idx += fixed_cA_len

        if len(cA_resampled) == cA_len:
            cA_original_scale = cA_resampled
        else:
            x = np.linspace(0, 1, fixed_cA_len)
            f_interp = interp1d(x, cA_resampled, kind='linear', fill_value='extrapolate')
            cA_original_scale = f_interp(np.linspace(0, 1, cA_len))

        # Reconstruct using approximation coefficients only
        cDs = [np.zeros(l) for l in detail_lens]
        coeffs = [cA_original_scale] + cDs
        rec = pywt.waverec(coeffs, wavelet)

        # Match original sequence length
        rec = rec[:orig_len] if rec.size > orig_len else np.pad(rec, (0, orig_len - rec.size))
        recon_features.append(rec)

    return np.vstack(recon_features).T

def dwt_to_time(recon_dwt, all_metas, num_features, fixed_cA_len=256):
    """Convert DWT features back to variable-length time-domain sequences."""
    if isinstance(recon_dwt, torch.Tensor):
        recon_dwt = recon_dwt.detach().cpu().numpy()
    
    reconstructed = []
    for i, meta in enumerate(all_metas):
        feat_vec = recon_dwt[i]
        recon_seq = unresample_and_reconstruct_single(feat_vec, meta, fixed_cA_len)
        reconstructed.append(recon_seq)
    return reconstructed

# -------------------------
# FFT Preprocessing Functions
# -------------------------

def sequences_to_fft(sequences, fixed_len=256):
    """Convert list of numpy sequences to FFT features (resampled real and imaginary parts)"""
    fft_features = []
    all_metas = []
    if len(sequences) == 0:
        return torch.FloatTensor([]), [], 0
    num_features = sequences[0].shape[1]
    
    for seq in sequences:
        T, F = seq.shape
        feat_parts = []
        feature_meta = []
        
        for f in range(F):
            sig = seq[:, f]
            # Compute FFT
            fft_coeffs = np.fft.rfft(sig)
            # Store real and imaginary parts separately
            real_part = np.real(fft_coeffs)
            imag_part = np.imag(fft_coeffs)
            
            # Resample both parts to fixed length
            real_resampled = resample_coeffs(real_part, fixed_len)
            imag_resampled = resample_coeffs(imag_part, fixed_len)
            
            feat_parts.extend([real_resampled, imag_resampled])
            feature_meta.append({
                'orig_len': T,
                'fft_len': len(fft_coeffs)
            })
        
        feat_vec = np.concatenate(feat_parts)
        fft_features.append(feat_vec)
        all_metas.append({'num_features': F, 'feature_meta': feature_meta})
    
    # FFT mode: each feature produces real + imaginary, so 2x features
    processed_num_features = num_features * 2
    return torch.FloatTensor(np.array(fft_features)), all_metas, processed_num_features

def fft_to_time(recon_fft, all_metas, num_features, fixed_len=256):
    """Convert reconstructed FFT features back to time domain sequences"""
    if isinstance(recon_fft, torch.Tensor):
        recon_fft = recon_fft.detach().cpu().numpy()
    reconstructed = []
    
    for i, meta in enumerate(all_metas):
        feat_vec = recon_fft[i]
        F = meta['num_features']
        recon_features = []
        idx = 0
        
        for f in range(F):
            fm = meta['feature_meta'][f]
            orig_len = fm['orig_len']
            fft_len = fm['fft_len']
            
            # Extract resampled real and imaginary parts
            real_resampled = feat_vec[idx:idx + fixed_len]
            imag_resampled = feat_vec[idx + fixed_len:idx + 2 * fixed_len]
            idx += 2 * fixed_len
            
            # Unresample back to original FFT length
            if fixed_len == fft_len:
                real_orig = real_resampled
                imag_orig = imag_resampled
            else:
                x = np.linspace(0, 1, fixed_len)
                real_interp = interp1d(x, real_resampled, kind='linear', fill_value='extrapolate')
                imag_interp = interp1d(x, imag_resampled, kind='linear', fill_value='extrapolate')
                real_orig = real_interp(np.linspace(0, 1, fft_len))
                imag_orig = imag_interp(np.linspace(0, 1, fft_len))
            
            # Reconstruct complex coefficients and apply inverse FFT
            fft_coeffs = real_orig + 1j * imag_orig
            rec = np.fft.irfft(fft_coeffs, n=orig_len)
            recon_features.append(rec)
        
        recon = np.vstack(recon_features).T
        reconstructed.append(recon)
    
    return reconstructed

# -------------------------
# Raw Resampling Functions
# -------------------------

def sequences_to_raw_resample(sequences, fixed_len=256):
    """Resample raw sequences directly to fixed length"""
    resampled_features = []
    all_metas = []
    if len(sequences) == 0:
        return torch.FloatTensor([]), [], 0
    num_features = sequences[0].shape[1]
    
    for seq in sequences:
        T, F = seq.shape
        resampled_parts = []
        
        for f in range(F):
            sig = seq[:, f]
            resampled_sig = resample_coeffs(sig, fixed_len)
            resampled_parts.append(resampled_sig)
        
        # Concatenate feature-first to match DWT organization: [f0_all, f1_all, ...]
        feat_vec = np.concatenate(resampled_parts)
        resampled_features.append(feat_vec)
        all_metas.append({'orig_len': T, 'num_features': F})
    
    return torch.FloatTensor(np.array(resampled_features)), all_metas, num_features

def raw_resample_to_time(recon_resampled, all_metas, num_features, fixed_len=256):
    """Convert resampled sequences back to original time length"""
    if isinstance(recon_resampled, torch.Tensor):
        recon_resampled = recon_resampled.detach().cpu().numpy()
    reconstructed = []
    
    for i, meta in enumerate(all_metas):
        feat_vec = recon_resampled[i]
        orig_len = meta['orig_len']
        F = meta['num_features']
        
        # Extract features from feature-first organization: [f0_all, f1_all, ...]
        recon_features = []
        for f in range(F):
            sig_resampled = feat_vec[f * fixed_len:(f + 1) * fixed_len]
            
            if fixed_len == orig_len:
                sig_orig = sig_resampled
            else:
                x = np.linspace(0, 1, fixed_len)
                f_interp = interp1d(x, sig_resampled, kind='linear', fill_value='extrapolate')
                sig_orig = f_interp(np.linspace(0, 1, orig_len))
            
            recon_features.append(sig_orig)
        
        # Stack as [T, F]
        recon_seq = np.vstack(recon_features).T
        reconstructed.append(recon_seq)
    
    return reconstructed

# -------------------------
# FFT2 Low-Pass Truncation Functions
# -------------------------

def sequences_to_fft2(sequences, fixed_len=256):
    """
    Convert sequences to low-pass filtered FFT representation.
    Retains only the lowest 'fixed_len' frequency components (including DC).
    Uses real and imaginary parts separately to preserve phase information.
    """
    fft_features = []
    all_metas = []
    if len(sequences) == 0:
        return torch.FloatTensor([]), [], 0
    num_features = sequences[0].shape[1]
    
    for seq in sequences:
        T, F = seq.shape
        feat_parts = []
        feature_meta = []
        
        for f in range(F):
            sig = seq[:, f]
            
            # Apply real FFT
            fft_vals = np.fft.rfft(sig)
            orig_fft_len = len(fft_vals)
            
            # Truncate or zero-pad to fixed_len
            if orig_fft_len >= fixed_len:
                # Keep only lowest 'fixed_len' frequencies (low-pass)
                fft_truncated = fft_vals[:fixed_len]
            else:
                # Zero-pad if fewer than fixed_len frequencies
                fft_truncated = np.zeros(fixed_len, dtype=complex)
                fft_truncated[:orig_fft_len] = fft_vals
            
            # Separate real and imaginary parts to preserve phase
            fft_real = np.real(fft_truncated)
            fft_imag = np.imag(fft_truncated)
            
            feat_parts.extend([fft_real, fft_imag])
            
            feature_meta.append({
                'orig_len': T,
                'orig_fft_len': orig_fft_len
            })
        
        # Concatenate: [f0_real, f0_imag, f1_real, f1_imag, ...]
        feat_vec = np.concatenate(feat_parts)
        fft_features.append(feat_vec)
        all_metas.append({'num_features': F, 'feature_meta': feature_meta})
    
    # FFT2 mode: each feature produces real + imaginary, so 2x features
    processed_num_features = num_features * 2
    return torch.FloatTensor(np.array(fft_features)), all_metas, processed_num_features

def fft2_to_time(recon_fft2, all_metas, num_features, fixed_len=256):
    """
    Convert reconstructed FFT (real + imag) back to time domain.
    Zero-pads the truncated FFT back to original length and applies inverse rFFT.
    """
    if isinstance(recon_fft2, torch.Tensor):
        recon_fft2 = recon_fft2.detach().cpu().numpy()
    reconstructed = []
    
    for i, meta in enumerate(all_metas):
        feat_vec = recon_fft2[i]
        F = meta['num_features']
        recon_features = []
        
        for f in range(F):
            fm = meta['feature_meta'][f]
            orig_len = fm['orig_len']
            orig_fft_len = fm['orig_fft_len']
            
            # Extract real and imaginary parts for this feature
            idx_real = f * 2 * fixed_len
            idx_imag = f * 2 * fixed_len + fixed_len
            
            fft_real = feat_vec[idx_real:idx_real + fixed_len]
            fft_imag = feat_vec[idx_imag:idx_imag + fixed_len]
            
            # Reconstruct complex FFT
            fft_reconstructed_truncated = fft_real + 1j * fft_imag
            
            # Zero-pad back to original FFT length if needed
            if orig_fft_len > fixed_len:
                fft_reconstructed = np.zeros(orig_fft_len, dtype=complex)
                fft_reconstructed[:fixed_len] = fft_reconstructed_truncated
            else:
                fft_reconstructed = fft_reconstructed_truncated[:orig_fft_len]
            
            # Inverse rFFT to time domain
            sig_reconstructed = np.fft.irfft(fft_reconstructed, n=orig_len)
            recon_features.append(sig_reconstructed)
        
        recon_seq = np.vstack(recon_features).T
        reconstructed.append(recon_seq)
    
    return reconstructed

# -------------------------
# MLP VAE
# -------------------------

class DWTMLPVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64, hidden_dims=[1024, 512, 256]):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Learnable direction vector for conditional prior (Brain Aging style)
        self.u = nn.Parameter(torch.randn(latent_dim, 1) * 0.1)  # Direction vector for TLX
        self.prior_sigma = nn.Parameter(torch.ones(1) * 1.0)  # Learnable prior variance

        # Encoder
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.Dropout(0.1))
            in_dim = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # TLX regressor from encoder features
        self.fc_y_mu = nn.Linear(hidden_dims[-1], 1)
        self.fc_y_logvar = nn.Linear(hidden_dims[-1], 1)

        # Decoder with residual connections
        hidden_dims_rev = list(reversed(hidden_dims))
        self.decoder_input = nn.Linear(latent_dim, hidden_dims_rev[0])
        
        # Decoder layers with residual connections
        self.decoder_layers = nn.ModuleList()
        in_dim = hidden_dims_rev[0]
        for h in hidden_dims_rev[1:]:
            self.decoder_layers.append(nn.Sequential(
                nn.Linear(in_dim, h),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.1)
            ))
            in_dim = h
        
        self.decoder_output = nn.Linear(hidden_dims_rev[-1], input_dim)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, h  # Return h for regressor

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def conditional_kl_divergence(self, mu, logvar, tlx_targets):
        """
        Compute KL divergence between q(z|x) and conditional prior p(z|y)
        p(z|y) = N(u * y, σ²I) where u is learnable direction vector
        """
        # Conditional prior mean: μ_prior = u * y (element-wise broadcast)
        # u: [latent_dim, 1], tlx_targets: [batch]
        mu_prior = self.u.squeeze(-1).unsqueeze(0) * tlx_targets.unsqueeze(1)  # [batch, latent_dim]
        
        # Conditional prior variance (learnable)
        sigma_prior_sq = torch.clamp(self.prior_sigma ** 2, min=1e-6)
        
        # KL divergence: D_KL(q(z|x) || p(z|y))
        var_q = torch.exp(logvar)  # q(z|x) variance
        kl_div = 0.5 * torch.sum(
            torch.log(sigma_prior_sq) - logvar +
            (var_q + (mu - mu_prior) ** 2) / sigma_prior_sq - 1,
            dim=1
        )
        return torch.mean(kl_div)

    def decode(self, z):
        h = self.decoder_input(z)
        h = F.leaky_relu(h, 0.01)
        
        for layer in self.decoder_layers:
            h_new = layer(h)
            # Add residual connection if dimensions match
            h = h + h_new if h.shape[-1] == h_new.shape[-1] else h_new
        
        return self.decoder_output(h)

    def forward(self, x):
        mu, logvar, h = self.encode(x)
        
        # VAE reconstruction
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        # TLX regression
        y_mu = self.fc_y_mu(h)
        y_logvar = self.fc_y_logvar(h)
        
        return recon, mu, logvar, y_mu, y_logvar

# -------------------------
# Convolutional VAE for DWT coefficients
# -------------------------

class DWTConvVAE(nn.Module):
    def __init__(self, num_features, fixed_cA_len, latent_dim=64):
        super().__init__()
        self.num_features = num_features
        self.fixed_cA_len = fixed_cA_len
        self.latent_dim = latent_dim
        
        # Conditional prior parameters for TLX regression
        self.u = nn.Parameter(torch.randn(latent_dim, 1) * 0.1)
        self.prior_sigma = nn.Parameter(torch.ones(1) * 1.0)
        
        # Conv1D encoder for DWT coefficients
        self.encoder = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
        )
        
        # Calculate the flattened size after convolutions
        self.conv_output_size = 256 * (fixed_cA_len // 8)
        
        # Latent space projection
        self.fc_mu = nn.Linear(self.conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.conv_output_size, latent_dim)
        
        # TLX regressor from flattened encoder features
        self.fc_y_mu = nn.Linear(self.conv_output_size, 1)
        self.fc_y_logvar = nn.Linear(self.conv_output_size, 1)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.conv_output_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            
            nn.ConvTranspose1d(64, num_features, kernel_size=7, stride=2, padding=3, output_padding=1),
        )
    
    def encode(self, x):
        x = x.view(-1, self.num_features, self.fixed_cA_len)
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar, h_flat
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def conditional_kl_divergence(self, mu, logvar, tlx_targets):
        """
        Compute KL divergence between q(z|x) and conditional prior p(z|y)
        p(z|y) = N(u * y, σ²I) where u is learnable direction vector
        """
        # Conditional prior mean: μ_prior = u * y (element-wise broadcast)
        # u: [latent_dim, 1], tlx_targets: [batch]
        mu_prior = self.u.squeeze(-1).unsqueeze(0) * tlx_targets.unsqueeze(1)  # [batch, latent_dim]
        
        # Conditional prior variance (learnable)
        sigma_prior_sq = torch.clamp(self.prior_sigma ** 2, min=1e-6)
        
        # KL divergence: D_KL(q(z|x) || p(z|y))
        var_q = torch.exp(logvar)  # q(z|x) variance
        kl_div = 0.5 * torch.sum(
            torch.log(sigma_prior_sq) - logvar +
            (var_q + (mu - mu_prior) ** 2) / sigma_prior_sq - 1,
            dim=1
        )
        return torch.mean(kl_div)

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 256, self.fixed_cA_len // 8)
        h = self.decoder(h)
        return h.view(h.size(0), -1)
    
    def forward(self, x):
        # Get encoder features (shared between VAE and regressor)
        mu, logvar, h_flat = self.encode(x)
        
        # VAE path: reparameterize and decode
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        # Regressor from encoder features h_flat (more stable)
        y_mu = self.fc_y_mu(h_flat)
        y_logvar = self.fc_y_logvar(h_flat)
        
        return recon, mu, logvar, y_mu, y_logvar

# -------------------------
# LSTM VAE for DWT coefficients
# -------------------------

class DWTLSTMVAE(nn.Module):
    def __init__(self, num_features, fixed_cA_len, latent_dim=64, hidden_size=256, num_layers=2):
        super().__init__()
        self.num_features = num_features
        self.fixed_cA_len = fixed_cA_len
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Conditional prior parameters for TLX regression
        self.u = nn.Parameter(torch.randn(latent_dim, 1) * 0.1)
        self.prior_sigma = nn.Parameter(torch.ones(1) * 1.0)
        
        # LSTM encoder: processes sequence of DWT coefficients
        # Input: [batch, seq_len, input_size] = [batch, fixed_cA_len, num_features]
        self.encoder_lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Latent space projection from final hidden state
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)
        
        # TLX regressor from encoder hidden state
        self.fc_y_mu = nn.Linear(hidden_size, 1)
        self.fc_y_logvar = nn.Linear(hidden_size, 1)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_size)
        
        # LSTM decoder: generates sequence from latent representation
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Output projection
        self.decoder_output = nn.Linear(hidden_size, num_features)
    
    def encode(self, x):
        # x: [batch, num_features * fixed_cA_len] -> reshape to [batch, fixed_cA_len, num_features]
        x = x.view(-1, self.num_features, self.fixed_cA_len)
        x = x.permute(0, 2, 1)  # [batch, fixed_cA_len, num_features]
        
        # LSTM forward pass
        _, (h_n, _) = self.encoder_lstm(x)  # h_n: [num_layers, batch, hidden_size]
        
        # Use final layer's hidden state
        h = h_n[-1]  # [batch, hidden_size]
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, h
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def conditional_kl_divergence(self, mu, logvar, tlx_targets):
        """
        Compute KL divergence between q(z|x) and conditional prior p(z|y)
        p(z|y) = N(u * y, σ²I) where u is learnable direction vector
        """
        # Conditional prior mean: μ_prior = u * y (element-wise broadcast)
        # u: [latent_dim, 1], tlx_targets: [batch]
        mu_prior = self.u.squeeze(-1).unsqueeze(0) * tlx_targets.unsqueeze(1)  # [batch, latent_dim]
        
        # Conditional prior variance (learnable)
        sigma_prior_sq = torch.clamp(self.prior_sigma ** 2, min=1e-6)
        
        # KL divergence: D_KL(q(z|x) || p(z|y))
        var_q = torch.exp(logvar)  # q(z|x) variance
        kl_div = 0.5 * torch.sum(
            torch.log(sigma_prior_sq) - logvar +
            (var_q + (mu - mu_prior) ** 2) / sigma_prior_sq - 1,
            dim=1
        )
        return torch.mean(kl_div)
    
    def decode(self, z):
        batch_size = z.size(0)
        
        # Initialize LSTM hidden state from latent vector
        h_0 = self.decoder_input(z)  # [batch, hidden_size]
        h_0 = h_0.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, batch, hidden_size]
        c_0 = torch.zeros_like(h_0)  # Initialize cell state to zeros
        
        # Feed zeros as input, let hidden state drive generation
        decoder_input = torch.zeros(batch_size, self.fixed_cA_len, self.hidden_size, device=z.device)
        
        # LSTM decoder with initialized hidden state
        h, _ = self.decoder_lstm(decoder_input, (h_0, c_0))  # [batch, fixed_cA_len, hidden_size]
        
        # Project to output
        h = self.decoder_output(h)  # [batch, fixed_cA_len, num_features]
        
        # Permute and flatten back to original format
        h = h.permute(0, 2, 1)  # [batch, num_features, fixed_cA_len]
        h = h.reshape(batch_size, -1)  # [batch, num_features * fixed_cA_len]
        
        return h
    
    def forward(self, x):
        mu, logvar, h = self.encode(x)
        
        # VAE reconstruction
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        # TLX regression from encoder hidden state
        y_mu = self.fc_y_mu(h)
        y_logvar = self.fc_y_logvar(h)
        
        return recon, mu, logvar, y_mu, y_logvar

# -------------------------
# Training wrapper
# -------------------------

class DWTAVAERTrainer(nn.Module):
    def __init__(self, fixed_cA_len=256, latent_dim=64, hidden_dims=[1024,512,256], lr=1e-3, batch_size=32, epochs=50, wavelet='db4', level=4, device='cuda', use_time_loss=False, architecture='mlp', rho=0.5, annealing=None, max_weight=0.5, num_cycles=10, mode='dwt'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.fixed_cA_len = fixed_cA_len
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.wavelet = wavelet
        self.level = level
        self.use_time_loss = use_time_loss
        self.architecture = architecture  # 'mlp', 'conv', or 'lstm'
        self.rho = rho  # AVAE correlation parameter (0 < rho < 1)
        self.annealing = annealing  # Annealing strategy: None, "monotonic", or "cyclic"
        self.max_weight = max_weight  # Maximum weight for annealing
        self.num_cycles = num_cycles  # Number of cycles for cyclic annealing
        self.mode = mode  # Preprocessing mode: 'dwt', 'fft', or 'raw'
        self.tlx_mean = None
        self.tlx_std = None
    
    def get_annealing_weight(self, epoch):
        """Calculate annealing weight for KL, AVAE, and regression losses.
        
        Monotonic: First 50% of epochs for annealing (0.01 → max_weight), rest at max_weight
        Cyclic: Each cycle uses 50% for annealing (0.01 → max_weight), 50% at max_weight
        """
        if self.annealing is None:
            # No annealing - all weights at 1.0
            return 1.0
        elif self.annealing == "monotonic":
            # First 50% of epochs for annealing, rest at max_weight
            annealing_epochs = self.epochs // 2
            
            if epoch < annealing_epochs:
                # Annealing phase: linear increase from 0.01 to max_weight
                progress = epoch / annealing_epochs
                return 0.01 + progress * (self.max_weight - 0.01)
            else:
                # Plateau phase: stay at max_weight
                return self.max_weight
        elif self.annealing == "cyclic":
            # Each cycle: 50% annealing, 50% at max_weight
            cycle_length = self.epochs // self.num_cycles
            annealing_epochs = cycle_length // 2
            cycle_pos = epoch % cycle_length
            
            if cycle_pos < annealing_epochs:
                # Annealing phase: linear increase from 0.01 to max_weight
                progress = cycle_pos / annealing_epochs
                return 0.01 + progress * (self.max_weight - 0.01)
            else:
                # Plateau phase: stay at max_weight
                return self.max_weight
        else:
            raise ValueError(f"Unknown annealing strategy: {self.annealing}")

    def fit(self, padded_data, tlx_targets=None):
        # extract_variable_length_sequences must return list of numpy arrays [seq_len, num_features]
        sequences = extract_variable_length_sequences(padded_data)

        # Normalize sequences individually (zero mean, unit variance)
        sequences_norm = []
        seq_means = []
        seq_stds = []
        for seq in sequences:
            seq_np = np.asarray(seq, dtype=np.float32)
            mu = seq_np.mean(axis=0, keepdims=True)
            sigma = seq_np.std(axis=0, keepdims=True) + 1e-7
            seq_norm = (seq_np - mu) / sigma
            sequences_norm.append(seq_norm)
            seq_means.append(mu)
            seq_stds.append(sigma)
        
        # Normalize TLX targets
        tlx_normalized = None
        if tlx_targets is not None:
            tlx_targets = np.array(tlx_targets, dtype=np.float32)
            self.tlx_mean = np.mean(tlx_targets)
            self.tlx_std = np.std(tlx_targets) + 1e-7
            tlx_normalized = (tlx_targets - self.tlx_mean) / self.tlx_std
            # print(f"TLX normalization: mean={self.tlx_mean:.2f}, std={self.tlx_std:.2f}")
        
        # Apply preprocessing based on mode
        if self.mode == 'dwt':
            preprocessed_data, all_metas, num_features = sequences_to_dwt(
                sequences_norm, self.fixed_cA_len, self.wavelet, self.level
            )
        elif self.mode == 'fft':
            preprocessed_data, all_metas, num_features = sequences_to_fft(
                sequences_norm, self.fixed_cA_len
            )
        elif self.mode == 'fft2':
            preprocessed_data, all_metas, num_features = sequences_to_fft2(
                sequences_norm, self.fixed_cA_len
            )
        elif self.mode == 'raw':
            preprocessed_data, all_metas, num_features = sequences_to_raw_resample(
                sequences_norm, self.fixed_cA_len
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Choose 'dwt', 'fft', 'fft2', or 'raw'.")
        
        self.num_features = num_features
        self.all_metas = all_metas

        input_dim = preprocessed_data.shape[1]

        # Determine number of channels for Conv architecture
        if self.mode == 'fft' or self.mode == 'fft2':
            # FFT and FFT2 modes: 2 values (real + imag) per feature
            num_channels = num_features * 2
        else:
            # DWT, Raw modes: 1 value per feature
            num_channels = num_features

        # Initialize model architecture
        if self.architecture == 'conv':
            self.model = DWTConvVAE(num_channels, self.fixed_cA_len, self.latent_dim).to(self.device)
            print(f"Using Conv1D AVAER with {num_channels} channels and {self.fixed_cA_len} coefficients")
        elif self.architecture == 'lstm':
            self.model = DWTLSTMVAE(num_channels, self.fixed_cA_len, self.latent_dim).to(self.device)
            print(f"Using LSTM AVAER with {num_channels} channels and {self.fixed_cA_len} coefficients")
        else:
            self.model = DWTMLPVAE(input_dim, self.latent_dim, self.hidden_dims).to(self.device)
            print(f"Using MLP AVAER with input_dim={input_dim}")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Prepare DataLoader
        idxs = torch.arange(preprocessed_data.shape[0], dtype=torch.long)
        if tlx_normalized is not None:
            tlx_tensor = torch.FloatTensor(tlx_normalized)
            dataset = TensorDataset(preprocessed_data, idxs, tlx_tensor)
        else:
            dataset = TensorDataset(preprocessed_data, idxs)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.model.train()
        hist = []

        for epoch in range(self.epochs):
            total_loss = 0.0
            total_recon = 0.0
            total_kl = 0.0
            total_avae = 0.0
            total_reg = 0.0
            
            # Get annealing weight for this epoch
            beta = self.get_annealing_weight(epoch)
            
            for batch_data in loader:
                if tlx_normalized is not None:
                    batch_dwt, batch_idxs, batch_tlx = batch_data
                    batch_tlx = batch_tlx.to(self.device)
                else:
                    batch_dwt, batch_idxs = batch_data
                    batch_tlx = None
                    
                batch_dwt = batch_dwt.to(self.device)
                optimizer.zero_grad()
                
                # Step 1: Generate auxiliary samples (with gradient stopped)
                with torch.no_grad():
                    aux_dwt, _, _, _, _ = self.model(batch_dwt)
                
                # Step 2: Standard VAE forward pass on original data
                recon_dwt, mu, logvar, y_mu, y_logvar = self.model(batch_dwt)

                # Reconstruction loss (no annealing)
                recon_loss = F.mse_loss(recon_dwt, batch_dwt, reduction='mean')

                # KL divergence loss
                if tlx_normalized is not None and batch_tlx is not None:
                    kl_loss = self.model.conditional_kl_divergence(mu, logvar, batch_tlx)
                else:
                    kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1))

                # Step 3: Encode auxiliary samples
                mu_aux, logvar_aux, _ = self.model.encode(aux_dwt)
                
                # Convert to variance
                var = torch.exp(logvar)
                var_aux = torch.exp(logvar_aux)
                
                # Step 4: AVAE-specific terms (following DeepMind implementation exactly)
                # Expected log conditional: E[log p(z_aux | z)] where p(z_aux | z) = N(rho * z, (1 - rho^2) * I)
                
                # Compute the quadratic form in the exponent (per sample)
                numerator = (
                    var_aux + 
                    (self.rho ** 2) * var + 
                    torch.square(mu_aux - self.rho * mu)
                )
                # Sum over latent dimensions to get per-sample quadratic terms
                quadratic_term = torch.sum(numerator, dim=1) / (2.0 * (1.0 - self.rho ** 2))
                
                # Log normalizer (constant per sample)
                latent_dim = mu.shape[1]
                log_normalizer = -0.5 * latent_dim * np.log(2 * np.pi * (1.0 - self.rho ** 2))
                
                # Expected log conditional per sample, then mean over batch
                expected_log_conditional = torch.mean(log_normalizer - quadratic_term)
                
                # Entropy of auxiliary latent: H(q(z_aux | x_aux))
                # Sum over latent dimensions to get per-sample entropy, then mean over batch
                entropy_aux = torch.mean(torch.sum(0.5 * torch.log(2 * np.pi * np.e * var_aux), dim=1))
                
                # AVAE loss term (regularization): E[log p(z_aux|z)] + H(z_aux)
                # This should be maximized (or its negative minimized)
                avae_term = expected_log_conditional + entropy_aux

                # TLX regression loss (with numerical stability)
                reg_loss = torch.tensor(0.0, device=self.device)
                if tlx_normalized is not None and batch_tlx is not None:
                    # Clamp logvar to reasonable range: log(variance) in [-5, 5] 
                    # means variance in [0.0067, 148.4], std in [0.082, 12.2]
                    y_logvar_clamped = torch.clamp(y_logvar, min=-5, max=5)
                    # NLL for Gaussian: 0.5 * (log(2πσ²) + (y-μ)²/σ²)
                    # = 0.5 * (log(2π) + log(σ²) + (y-μ)²/σ²)
                    # = 0.5 * (log(2π) + logvar + (y-μ)²/exp(logvar))
                    mse = (y_mu.squeeze() - batch_tlx) ** 2
                    precision = torch.exp(-y_logvar_clamped)  # 1/σ²
                    reg_loss = 0.5 * torch.mean(y_logvar_clamped + mse * precision)

                # Total loss: Standard VAE + AVAE regularization + Regression
                # Minimize: recon + β*KL - β*AVAE + β*reg
                # All losses except reconstruction are annealed with β
                if self.annealing:
                    # Apply same beta to KL, AVAE, and regression losses
                    # Note: AVAE term is SUBTRACTED (we want to maximize it)
                    loss = (recon_loss + 
                           beta * (kl_loss - avae_term) + 
                           beta * reg_loss)
                else:
                    # No annealing: full weight on all terms
                    lambda_reg = 1.0 if tlx_normalized is not None else 0.0
                    loss = (recon_loss + 
                           kl_loss - 
                           avae_term + 
                           lambda_reg * reg_loss)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                optimizer.step()

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
                total_avae += avae_term.item()
                total_reg += reg_loss.item()

            avg_loss = total_loss / len(loader)
            avg_recon = total_recon / len(loader)
            avg_kl = total_kl / len(loader)
            avg_avae = total_avae / len(loader)
            avg_reg = total_reg / len(loader)
            
            hist.append(avg_loss)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                if self.annealing:
                    print(f"Epoch {epoch+1:4d}/{self.epochs} | Loss: {avg_loss:.4f} | R: {avg_recon:.4f} | KL: {avg_kl:.4f} | Aux: {avg_avae:.4f} | Reg: {avg_reg:.4f} | β={beta:.4f}")
                else:
                    print(f"Epoch {epoch+1:4d}/{self.epochs} | Loss: {avg_loss:.4f} | R: {avg_recon:.4f} | KL: {avg_kl:.4f} | Aux: {avg_avae:.4f} | Reg: {avg_reg:.4f}")

        # Store normalization parameters for inference
        self.sequences = sequences
        self.sequences_norm = sequences_norm
        self.seq_means = seq_means
        self.seq_stds = seq_stds
        return {'total_loss': hist}

    def encode(self, padded_data):
        sequences = extract_variable_length_sequences(padded_data)
        sequences_norm = []
        for seq in sequences:
            seq_np = np.asarray(seq, dtype=np.float32)
            mu = seq_np.mean(axis=0, keepdims=True)
            sigma = seq_np.std(axis=0, keepdims=True) + 1e-7
            sequences_norm.append((seq_np - mu) / sigma)
        
        # Apply preprocessing based on mode
        if self.mode == 'dwt':
            preprocessed_data, _, _ = sequences_to_dwt(sequences_norm, self.fixed_cA_len, self.wavelet, self.level)
        elif self.mode == 'fft':
            preprocessed_data, _, _ = sequences_to_fft(sequences_norm, self.fixed_cA_len)
        elif self.mode == 'fft2':
            preprocessed_data, _, _ = sequences_to_fft2(sequences_norm, self.fixed_cA_len)
        elif self.mode == 'raw':
            preprocessed_data, _, _ = sequences_to_raw_resample(sequences_norm, self.fixed_cA_len)
        
        self.model.eval()
        with torch.no_grad():
            mu, logvar, _ = self.model.encode(preprocessed_data.to(self.device))
        return mu.cpu().numpy()

    def transform(self, dataset_or_data):
        if hasattr(dataset_or_data, 'tensors'):
            padded_data = dataset_or_data.tensors[0].numpy()
        else:
            padded_data = dataset_or_data
        return self.encode(padded_data)

    def reconstruct(self, dataset_or_data):
        if hasattr(dataset_or_data, 'tensors'):
            padded_data = dataset_or_data.tensors[0].numpy()
        else:
            padded_data = dataset_or_data

        sequences = extract_variable_length_sequences(padded_data)
        sequences_norm = []
        seq_means = []
        seq_stds = []
        for seq in sequences:
            seq_np = np.asarray(seq, dtype=np.float32)
            mu = seq_np.mean(axis=0, keepdims=True)
            sigma = seq_np.std(axis=0, keepdims=True) + 1e-7
            sequences_norm.append((seq_np - mu) / sigma)
            seq_means.append(mu)
            seq_stds.append(sigma)

        # Apply preprocessing based on mode
        if self.mode == 'dwt':
            preprocessed_data, metas, _ = sequences_to_dwt(sequences_norm, self.fixed_cA_len, self.wavelet, self.level)
        elif self.mode == 'fft':
            preprocessed_data, metas, _ = sequences_to_fft(sequences_norm, self.fixed_cA_len)
        elif self.mode == 'fft2':
            preprocessed_data, metas, _ = sequences_to_fft2(sequences_norm, self.fixed_cA_len)
        elif self.mode == 'raw':
            preprocessed_data, metas, _ = sequences_to_raw_resample(sequences_norm, self.fixed_cA_len)
        
        self.model.eval()
        with torch.no_grad():
            recon_preprocessed, _, _, _, _ = self.model(preprocessed_data.to(self.device))
        
        # Reconstruct normalized time-domain sequences based on mode
        if self.mode == 'dwt':
            recon_norm = dwt_to_time(recon_preprocessed, metas, self.num_features, self.fixed_cA_len)
        elif self.mode == 'fft':
            recon_norm = fft_to_time(recon_preprocessed, metas, self.num_features, self.fixed_cA_len)
        elif self.mode == 'fft2':
            recon_norm = fft2_to_time(recon_preprocessed, metas, self.num_features, self.fixed_cA_len)
        elif self.mode == 'raw':
            recon_norm = raw_resample_to_time(recon_preprocessed, metas, self.num_features, self.fixed_cA_len)
        
        # de-normalize back to original scale
        recon_denorm = []
        for rec, mu, sigma in zip(recon_norm, seq_means, seq_stds):
            rec = np.asarray(rec, dtype=np.float32)
            rec_denorm = rec * sigma + mu
            recon_denorm.append(rec_denorm)
        return recon_denorm
