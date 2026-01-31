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
# MLP VAE (Standard VAE without regression)
# -------------------------

class DWTMLPVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64, hidden_dims=[1024, 512, 256]):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.LeakyReLU(0.01))
            in_dim = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        hidden_dims_rev = list(reversed(hidden_dims))
        self.decoder_input = nn.Linear(latent_dim, hidden_dims_rev[0])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        in_dim = hidden_dims_rev[0]
        for h in hidden_dims_rev[1:]:
            self.decoder_layers.append(nn.Sequential(
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.01)
            ))
            in_dim = h
        
        self.decoder_output = nn.Linear(hidden_dims_rev[-1], input_dim)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = F.leaky_relu(h, 0.01)
        
        for layer in self.decoder_layers:
            h = layer(h)
        
        return self.decoder_output(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# -------------------------
# Convolutional VAE for DWT coefficients (Standard VAE without regression)
# -------------------------

class DWTConvVAE(nn.Module):
    def __init__(self, num_features, fixed_cA_len, latent_dim=64):
        super().__init__()
        self.num_features = num_features
        self.fixed_cA_len = fixed_cA_len
        self.latent_dim = latent_dim
        
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
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 256, self.fixed_cA_len // 8)
        h = self.decoder(h)
        return h.view(h.size(0), -1)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# -------------------------
# LSTM VAE for DWT coefficients (Standard VAE without regression)
# -------------------------

class DWTLSTMVAE(nn.Module):
    def __init__(self, num_features, fixed_cA_len, latent_dim=64, hidden_size=256, num_layers=2):
        super().__init__()
        self.num_features = num_features
        self.fixed_cA_len = fixed_cA_len
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
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
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
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
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# -------------------------
# Training wrapper
# -------------------------

class DWTAVAETrainer(nn.Module):
    def __init__(self, fixed_cA_len=256, latent_dim=64, hidden_dims=[1024,512,256], lr=1e-3, batch_size=32, epochs=50, wavelet='db4', level=4, device='cuda', use_time_loss=False, architecture='mlp', rho=0.5, annealing=None, max_weight=0.5, num_cycles=10):
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
    
    def get_annealing_weight(self, epoch):
        """Calculate annealing weight for KL and AVAE losses.
        
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
            # Example: 900 epochs, 3 cycles → 300 epochs/cycle (150 anneal + 150 plateau)
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
    
    def fit(self, padded_data):
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
        
        # Convert sequences to DWT features
        dwt_data, all_metas, num_features = sequences_to_dwt(sequences_norm, self.fixed_cA_len, self.wavelet, self.level)
        self.num_features = num_features
        self.all_metas = all_metas

        input_dim = dwt_data.shape[1]

        # Initialize model architecture
        if self.architecture == 'conv':
            self.model = DWTConvVAE(num_features, self.fixed_cA_len, self.latent_dim).to(self.device)
            print(f"Using Conv1D AVAE with {num_features} features and {self.fixed_cA_len} coefficients")
        elif self.architecture == 'lstm':
            self.model = DWTLSTMVAE(num_features, self.fixed_cA_len, self.latent_dim).to(self.device)
            print(f"Using LSTM AVAE with {num_features} features and {self.fixed_cA_len} coefficients")
        else:  # default to mlp
            self.model = DWTMLPVAE(input_dim, self.latent_dim, self.hidden_dims).to(self.device)
            print(f"Using MLP AVAE with input_dim={input_dim}")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Prepare DataLoader
        idxs = torch.arange(dwt_data.shape[0], dtype=torch.long)
        dataset = TensorDataset(dwt_data, idxs)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.model.train()
        hist = []

        for epoch in range(self.epochs):
            total_loss = 0.0
            total_recon = 0.0
            total_kl = 0.0
            total_avae = 0.0
            
            # Get annealing weight for this epoch
            beta = self.get_annealing_weight(epoch)
            
            for batch_dwt, batch_idxs in loader:
                batch_dwt = batch_dwt.to(self.device)
                optimizer.zero_grad()
                
                # Step 1: Generate auxiliary samples (with gradient stopped)
                with torch.no_grad():
                    aux_dwt, _, _ = self.model(batch_dwt)
                
                # Step 2: Standard VAE forward pass on original data
                recon_dwt, mu, logvar = self.model(batch_dwt)

                # Reconstruction loss (no annealing)
                recon_loss = F.mse_loss(recon_dwt, batch_dwt, reduction='mean')

                # KL divergence loss (standard VAE)
                kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1))

                # Step 3: Encode auxiliary samples
                mu_aux, logvar_aux = self.model.encode(aux_dwt)
                
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
                
                # Total loss: Standard VAE + AVAE regularization
                # Minimize: recon + β*KL - β*(E[log p] + H)
                # Both KL and AVAE terms are annealed with β to prevent drift
                if self.annealing:
                    # Reconstruction always at full weight
                    # Both KL and AVAE terms get annealed with same β
                    loss = recon_loss + beta * (kl_loss - avae_term)
                else:
                    # No annealing: full weight on all terms
                    loss = recon_loss + kl_loss - avae_term

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                optimizer.step()

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
                total_avae += avae_term.item()

            avg_loss = total_loss / len(loader)
            avg_recon = total_recon / len(loader)
            avg_kl = total_kl / len(loader)
            avg_avae = total_avae / len(loader)
            
            hist.append(avg_loss)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                if self.annealing:
                    print(f"Epoch {epoch+1:4d}/{self.epochs} | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | Aux: {avg_avae:.4f} | β={beta:.4f}")
                else:
                    print(f"Epoch {epoch+1:4d}/{self.epochs} | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | Aux: {avg_avae:.4f}")

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
        
        dwt_data, _, _ = sequences_to_dwt(sequences_norm, self.fixed_cA_len, self.wavelet, self.level)
        self.model.eval()
        with torch.no_grad():
            mu, logvar = self.model.encode(dwt_data.to(self.device))
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

        dwt_data, metas, _ = sequences_to_dwt(sequences_norm, self.fixed_cA_len, self.wavelet, self.level)
        self.model.eval()
        with torch.no_grad():
            recon_dwt, _, _ = self.model(dwt_data.to(self.device))
        
        # Convert back to time domain and denormalize
        recon_norm = dwt_to_time(recon_dwt, metas, self.num_features, self.fixed_cA_len)
        recon_denorm = []
        for rec, mu, sigma in zip(recon_norm, seq_means, seq_stds):
            rec = np.asarray(rec, dtype=np.float32)
            rec_denorm = rec * sigma + mu
            recon_denorm.append(rec_denorm)
        return recon_denorm
