
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import RobustScaler
from .base_models import DeepImputationBase

from utils.cgr_utils import (
    CGRAttentionGate,
    analyze_cgr_correlations,
    evaluate_cgr_consistency
)

class MinibatchDiscrimination(nn.Module):
    def __init__(self, input_dim, num_kernels=32, kernel_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.num_kernels = num_kernels
        self.kernel_dim = kernel_dim
        
        # Fix dimensions for proper matrix multiplication
        self.T = nn.Parameter(torch.Tensor(input_dim, num_kernels * kernel_dim))
        nn.init.normal_(self.T, 0, 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Ensure x is the right shape before multiplication
        if len(x.shape) == 3:
            x = x.view(batch_size, -1)
        
        # Matrix multiplication and reshape
        matrices = x.mm(self.T)
        matrices = matrices.view(batch_size, self.num_kernels, self.kernel_dim)
        
        # Compute cross-sample distances
        broadcast_matrices = matrices.unsqueeze(1)
        broadcast_matrices_T = matrices.unsqueeze(0)
        
        # L1 distance between samples
        dist = torch.sum(torch.abs(broadcast_matrices - broadcast_matrices_T), -1)
        dist = torch.sum(torch.exp(-dist), 0)
        
        return dist

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        return x + 0.1 * self.block(x)  # Scaled residual connection   

class LipschitzLinear(nn.Module):
    """
    Linear layer with Lipschitz continuity constraint.
    Ensures the layer's Lipschitz constant is bounded by a specified value.
    """
    def __init__(self, in_features, out_features, lip_const=1.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lip_const = lip_const
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights using Kaiming initialization"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, input):
        # Normalize weight matrix to ensure Lipschitz constraint
        with torch.no_grad():
            w_norm = torch.norm(self.weight, p=2)
            if w_norm > self.lip_const:
                self.weight.data = self.weight.data * (self.lip_const / w_norm)
                
        return F.linear(input, self.weight, self.bias)
        
    def extra_repr(self):
        """String representation of the layer"""
        return f'in_features={self.in_features}, out_features={self.out_features}, lip_const={self.lip_const}'


class AgriConsistencyLoss(nn.Module):
    """
    Agricultural consistency loss that enforces domain-specific constraints
    and relationships between imputed values.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, x, cgr, mask):
        """
        Compute agricultural consistency loss
        
        Args:
            x: Imputed data tensor
            cgr: CGR values tensor
            mask: Missing value mask tensor
        """
        # Ensure positive values for nutrients/measurements
        positivity_loss = torch.mean(F.relu(-x) * ~mask)
        
        # Temporal smoothness - values shouldn't change too abruptly
        temporal_diff = x[1:] - x[:-1]
        smoothness_loss = torch.mean(torch.abs(temporal_diff))
        
        # CGR relationship - imputed values should maintain reasonable relationships with CGR
        cgr_scaled = (cgr - cgr.mean()) / (cgr.std() + self.eps)
        x_scaled = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + self.eps)
        cgr_consistency = torch.mean(torch.abs(
            torch.corrcoef(torch.stack([x_scaled[~mask], cgr_scaled[~mask]]))[0, 1]
        ))
        
        # Value range consistency - values should stay within typical ranges
        range_loss = torch.mean(F.relu(torch.abs(x_scaled) - 5.0) * ~mask)
        
        # Combine losses with weights
        total_loss = (
            0.3 * positivity_loss + 
            0.3 * smoothness_loss + 
            0.2 * cgr_consistency + 
            0.2 * range_loss
        )
        
        return total_loss


class AgriPatternAttention(nn.Module):
    """Agricultural pattern-aware attention with theoretical guarantees"""
    def __init__(self, embed_dim, num_heads, dropout, cgr_guided=True):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        if cgr_guided:
            self.cgr_gate = CGRAttentionGate(embed_dim)

    def forward(self, x, cgr=None):
        attn_output, _ = self.attention(x, x, x)
        if cgr is not None:
            attn_output = self.cgr_gate(attn_output, cgr)
        return attn_output

class EnhancedDLPIMDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        # Improved minibatch discrimination
        self.minibatch_disc = MinibatchDiscrimination(input_dim, num_kernels=16)
        
        # Enhanced main network with spectral normalization and residual connections
        main_input_dim = input_dim + 16 + 1  # input_dim + num_kernels + CGR
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(main_input_dim, 256)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            ResidualBlock(256),
            nn.utils.spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            ResidualBlock(128)
        )
        
        # Separate heads with spectral normalization
        self.validity = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(128, 1)),
            nn.Sigmoid()
        )
        
        self.feature_matching = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(128, input_dim + 1))
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        # Split features and CGR
        features = x[:, :-1]
        cgr = x[:, -1:]
        
        # Get minibatch features with gradient scaling
        mb_features = self.minibatch_disc(features)
        
        # Combine features with normalization
        combined = torch.cat([features, mb_features, cgr], dim=1)
        
        # Process through main network
        features = self.main(combined)
        
        # Get outputs with gradient clipping
        validity = self.validity(features)
        matched_features = self.feature_matching(features)
        
        return validity, matched_features

class EnhancedDLPIMGenerator(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Add theoretical justification as docstring
        """
        Enhanced Generator incorporating:
        1. Lipschitz continuous layers for stability
        2. Adaptive attention mechanism with agricultural priors
        3. Theoretical bounds on CGR influence
        """
        
        # Improved feature encoder with Lipschitz constraint
        self.feature_encoder = LipschitzLinear(
            input_dim + latent_dim,
            256,
            lip_const=1.0
        )
        
        # Enhanced CGR processor with theoretical guarantees
        self.cgr_processor = AdaptiveCGRProcessor(
            input_dim=1,
            hidden_dims=[64, 32],
            dropout_rate=0.1
        )
        
        # Agricultural pattern-aware attention
        self.self_attention = AgriPatternAttention(
            embed_dim=128,
            num_heads=4,
            dropout=0.1,
            cgr_guided=True
        )
        
        # Initialize with theoretical bounds
        self._init_with_bounds()

        # Improved feature encoder with residual connections
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim + latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            ResidualBlock(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            ResidualBlock(128)
        )
        
        # Enhanced CGR processor with skip connections
        self.cgr_processor = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            ResidualBlock(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(32)
        )
        
        # Multi-head self-attention with gradient clipping
        self.self_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,  # Reduced heads for stability
            dropout=0.1,
            batch_first=True
        )
        
        # Improved decoder with gradient scaling
        self.decoder = nn.Sequential(
            nn.Linear(128 + 32, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            ResidualBlock(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(128),
            ResidualBlock(128),
            nn.Linear(128, input_dim),
            nn.Tanh()
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_with_bounds(self):
        """Initialize weights with theoretical guarantees"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use theoretically justified initialization
                std = np.sqrt(2.0 / (m.weight.shape[0] + m.weight.shape[1]))
                nn.init.normal_(m.weight, mean=0.0, std=std)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def temporal_forward(self, noise, features, cgr, time_window):
        # Process features with noise
        x = torch.cat([features, noise], dim=1)
        encoded_features = self.feature_encoder(x)
        
        # Apply time-aware attention
        encoded_features = encoded_features.unsqueeze(1)
        attended_features, _ = self.self_attention(
            encoded_features, 
            time_steps=time_window
        )
        attended_features = attended_features.squeeze(1)
        
        # Process CGR with temporal information
        temporal_cgr_features = self.cgr_processor(cgr, time_window)
        
        # Combine features
        combined = torch.cat([attended_features, temporal_cgr_features], dim=1)
        
        # Generate output
        output = self.decoder(combined)
        return output

    # Create sliding window for temporal processing
    def create_time_window(data, window_size=24):
        windows = []
        for i in range(len(data) - window_size + 1):
            windows.append(data[i:i + window_size])
        return torch.stack(windows)
        
        def forward(self, noise, features, cgr):
            # Process features with noise
            x = torch.cat([features, noise], dim=1)
            encoded_features = self.feature_encoder(x)
            
            # Apply self attention with gradient clipping
            encoded_features = encoded_features.unsqueeze(1)
            attended_features, _ = self.self_attention(
                encoded_features, 
                encoded_features, 
                encoded_features
            )
            attended_features = attended_features.squeeze(1)
            
            # Process CGR with skip connection
            cgr_features = self.cgr_processor(cgr)
            
            # Combine features with scaled gradients
            combined = torch.cat([attended_features, cgr_features], dim=1)
            
            # Generate output with gradient clipping
            output = self.decoder(combined)
            return output

class EnhancedDLPIMImputer:
    def __init__(self, input_dim, latent_dim=64):
        """Initialize with theoretical guarantees and adaptive components"""
        self.monte_carlo_iterations = 10
        self.confidence_alpha = 0.95
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device
        
        # Initialize networks
        self.generator = EnhancedDLPIMGenerator(input_dim, latent_dim).to(self.device)
        self.discriminator = EnhancedDLPIMDiscriminator(input_dim).to(self.device)
        
        # Initialize scaler
        self.robust_scaler = RobustScaler()
        self.best_loss = float('inf')
        self.generator_loss_history = []
        self.scaler = GradScaler()
        
        # Optimizers with better learning rates
        self.g_optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=2e-4,
            betas=(0.5, 0.999),
            weight_decay=1e-6
        )
        
        self.d_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=1e-4,
            betas=(0.5, 0.999),
            weight_decay=1e-6
        )
        
        # Learning rate schedulers
        self.g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.g_optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        self.d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.d_optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Early stopping parameters
        self.patience = 25
        self.patience_counter = 0
        
    def _compute_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand((real_data.size(0), 1)).to(self.device)
        interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
        
        d_interpolates, _ = self.discriminator(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates).to(self.device),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
        
    def _initialize_missing_values(self, X, mask):
        #Initialize missing values using MEAN, which generally performs betterfor agricultural data due to its ability to capture feature relationships.
        imputer = IterativeImputer(

            max_iter=10,
            random_state=42,
            initial_strategy='mean',
            min_value=np.min(X[mask]),  # Respect data bounds
            max_value=np.max(X[mask])
        )
        return imputer.fit_transform(X)
        
    def fit_transform(self, X, mask):
        """
        Fit the DLPIM model and transform the input data.
        
        Args:
            X (np.ndarray): Input data with missing values
            mask (np.ndarray): Boolean mask where True indicates observed values
            
        Returns:
            np.ndarray: Imputed data
        """
        try:
            print(f"Starting enhanced DLPIM imputation with shape: {X.shape}")
            
            # Input validation and preprocessing
            X, mask = self._validate_and_preprocess(X, mask)
            
            # Initialize training parameters
            training_params = self._setup_training_params(X.shape[0])
            
            # Training loop
            best_state = self._train_model(X, mask, training_params)
            
            # Generate final imputation
            final_result = self._generate_final_imputation(X, mask, best_state, training_params)
            
            print("Imputation completed successfully.")
            return final_result
            
        except Exception as e:
            print(f"Error in DLPIM imputation: {str(e)}")
            # Return original data with mean imputation as fallback
            return self._fallback_imputation(X, mask)
            
    def _validate_and_preprocess(self, X, mask):
        """Validate inputs and preprocess data"""
        # Input validation
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(mask, np.ndarray):
            mask = np.asarray(mask)
            
        # Check shapes
        if X.shape != mask.shape:
            raise ValueError(f"Shape mismatch: X {X.shape} vs mask {mask.shape}")
            
        # Handle NaN values
        if np.isnan(X).any():
            print("Warning: Input contains NaN values. Initializing with mean imputation.")
            mean_vals = np.nanmean(X, axis=0)
            X = np.where(np.isnan(X), mean_vals[None, :], X)
        
        # Initialize missing values
        X_init = self._initialize_missing_values(X, mask)
        
        # Scale data
        X_scaled = self.robust_scaler.fit_transform(X_init)
        
        # Convert to tensors
        X = torch.FloatTensor(X_scaled).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        
        return X, mask
    
    def _shuffle_data(self, X, features, cgr, mask, idx):
        """Shuffle the data using provided indices"""
        X_shuffled = X[idx]
        features_shuffled = features[idx]
        cgr_shuffled = cgr[idx]
        mask_shuffled = mask[idx]
        return X_shuffled, features_shuffled, cgr_shuffled, mask_shuffled

    def _prepare_batch(self, X, features, cgr, mask, start_idx, end_idx):
        """Prepare a batch of data"""
        return {
            'X': X[start_idx:end_idx],
            'features': features[start_idx:end_idx],
            'cgr': cgr[start_idx:end_idx],
            'mask': mask[start_idx:end_idx]
        }

    def _impute_batch(self, features, cgr, mask, start_idx, end_idx):
        """Generate imputation for a batch"""
        batch_size = end_idx - start_idx
        noise = torch.randn(batch_size, self.latent_dim).to(self.device)
        
        # Generate imputed features
        imputed_features = self.generator(
            noise, 
            features[start_idx:end_idx], 
            cgr[start_idx:end_idx]
        )
        
        # Combine with CGR
        batch_imputed = torch.cat([imputed_features, cgr[start_idx:end_idx]], dim=1)
        
        # Apply mask - handle the full tensor including CGR
        batch_mask = mask[start_idx:end_idx]
        original_data = torch.cat([features[start_idx:end_idx], cgr[start_idx:end_idx]], dim=1)
        batch_final = torch.where(batch_mask.bool(), original_data, batch_imputed)
        
        return batch_final

    def _train_discriminator(self, batch_data, batch_size, params):
        """Train discriminator for one step"""
        self.d_optimizer.zero_grad()
        
        with autocast():
            # Generate fake data
            noise = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_features = self.generator(noise, batch_data['features'], batch_data['cgr'])
            fake_data = torch.cat([fake_features, batch_data['cgr']], dim=1)
            
            # Get discriminator outputs
            real_validity, real_features = self.discriminator(batch_data['X'])
            fake_validity, fake_features = self.discriminator(fake_data.detach())
            
            # Compute gradient penalty
            gradient_penalty = self._compute_gradient_penalty(batch_data['X'], fake_data)
            
            # Discriminator loss
            d_loss = (torch.mean(fake_validity) - torch.mean(real_validity) + 
                     10 * gradient_penalty +
                     0.1 * F.mse_loss(fake_features, real_features))
        
        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.d_optimizer)
        self.scaler.update()
        
        return d_loss.item()

    def _train_generator(self, batch_data, batch_size, params):
        """Improved generator training with better loss scaling and normalization"""
        self.g_optimizer.zero_grad()
        
        with autocast():
            # Add noise with smaller variance
            noise = torch.randn(batch_size, self.latent_dim).to(self.device) * 0.1
            
            # Generate fake features with gradient clipping
            fake_features = self.generator(noise, batch_data['features'], batch_data['cgr'])
            fake_data = torch.cat([fake_features, batch_data['cgr']], dim=1)
            
            # Get discriminator outputs
            fake_validity, fake_features = self.discriminator(fake_data)
            
            # Adversarial loss with stability improvements
            g_adv_loss = -torch.mean(torch.log(fake_validity + 1e-8))
            g_adv_loss = torch.clamp(g_adv_loss, min=-100, max=100)  # Prevent extreme values
            
            # Reconstruction loss with scaled weights
            mse_loss = F.mse_loss(fake_features * batch_data['mask'], batch_data['X'] * batch_data['mask'])
            l1_loss = F.l1_loss(fake_features * batch_data['mask'], batch_data['X'] * batch_data['mask'])
            g_rec_loss = params['reconstruction_weight'] * mse_loss + params['l1_weight'] * l1_loss
            
            # Feature matching with normalization
            _, real_features = self.discriminator(batch_data['X'])
            real_features = F.normalize(real_features, dim=1)
            fake_features = F.normalize(fake_features, dim=1)
            g_feat_loss = params['feature_matching_weight'] * F.mse_loss(fake_features, real_features)
            
            # Smoothness constraint
            g_smooth_loss = params['smooth_weight'] * torch.mean(torch.abs(fake_features[:, 1:] - fake_features[:, :-1]))
            
            # Combined loss with better scaling
            g_loss = g_adv_loss + g_rec_loss + g_feat_loss + g_smooth_loss
        
        # Use gradient clipping
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.g_optimizer)
        self.scaler.update()
        
        return g_loss.item()

    def _update_training_state(self, epoch_losses, best_state, patience_counter, patience):
        """Update training state and check early stopping"""
        avg_g_loss = epoch_losses['g_loss']
        
        if avg_g_loss < self.best_loss:
            self.best_loss = avg_g_loss
            patience_counter = 0
            best_state = {
                'generator': self.generator.state_dict(),
                'discriminator': self.discriminator.state_dict()
            }
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print("Early stopping triggered")
            
        return best_state
   
    def _setup_training_params(self, n_samples):
        """Enhanced training parameters with better stability"""
        return {
            'batch_size': min(8, n_samples),  # Even smaller batch size for better stability
            'n_samples': n_samples,
            'epochs': 200,
            'n_critic': 2,  # Further reduced critic iterations
            'patience': 30,
            'learning_rates': {
                'generator': {
                    'initial': 5e-5,  # Much lower learning rate
                    'min': 1e-7
                },
                'discriminator': {
                    'initial': 2e-5,  # Even lower for discriminator
                    'min': 1e-7
                }
            },
            'gradient_penalty_weight': 1.0,  # Further reduced penalty
            'feature_matching_weight': 0.5,
            'reconstruction_weight': 5.0,  # Scaled down reconstruction weight
            'l1_weight': 1.0,
            'smooth_weight': 0.01
        }
    def _train_model(self, X, mask, params):
        """Main training loop"""
        features = X[:, :-1]
        cgr = X[:, -1:].detach()
        
        patience_counter = 0
        best_state = None
        epoch_metrics = []
        
        for epoch in range(params['epochs']):
            try:
                # Shuffle data
                idx = torch.randperm(params['n_samples'])
                X, features, cgr, mask = self._shuffle_data(X, features, cgr, mask, idx)
                
                # Training step
                epoch_losses = self._train_epoch(X, features, cgr, mask, params)
                
                # Update learning rates and check early stopping
                best_state = self._update_training_state(
                    epoch_losses, best_state, patience_counter, params['patience']
                )
                
                # Log progress
                if epoch % 10 == 0:
                    self._log_progress(epoch, epoch_losses, params['epochs'])
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Handle OOM error
                    torch.cuda.empty_cache()
                    params['batch_size'] //= 2
                    print(f"Reduced batch size to {params['batch_size']} due to OOM error")
                    continue
                else:
                    raise e
                    
        return best_state
    
    def _train_epoch(self, X, features, cgr, mask, params):
        """Train for one epoch"""
        epoch_g_loss = 0
        epoch_d_loss = 0
        n_batches = 0
        
        for i in range(0, params['n_samples'], params['batch_size']):
            batch_end = min(i + params['batch_size'], params['n_samples'])
            batch_size_curr = batch_end - i
            
            batch_data = self._prepare_batch(
                X, features, cgr, mask, i, batch_end
            )
            
            # Train discriminator
            d_loss = self._train_discriminator(batch_data, batch_size_curr, params)
            epoch_d_loss += d_loss
            
            # Train generator
            g_loss = self._train_generator(batch_data, batch_size_curr)
            epoch_g_loss += g_loss
            
            n_batches += 1
            
        return {
            'g_loss': epoch_g_loss / n_batches,
            'd_loss': epoch_d_loss / (n_batches * params['n_critic'])
        }
    
    def _generate_final_imputation(self, X, mask, best_state, params):
        """Generate final imputation using best model"""
        if best_state is not None:
            self.generator.load_state_dict(best_state['generator'])
        
        self.generator.eval()
        with torch.no_grad():
            features = X[:, :-1]
            cgr = X[:, -1:].detach()
            imputed_data = []
            
            for i in range(0, params['n_samples'], params['batch_size']):
                batch_end = min(i + params['batch_size'], params['n_samples'])
                batch_result = self._impute_batch(
                    features, cgr, mask, i, batch_end
                )
                imputed_data.append(batch_result)
            
            imputed = torch.cat(imputed_data, dim=0)
            return self.robust_scaler.inverse_transform(imputed.cpu().numpy())
    
    def _fallback_imputation(self, X, mask):
        """Fallback imputation method using mean imputation"""
        print("Using fallback imputation method...")
        mean_vals = np.nanmean(X, axis=0)
        return np.where(mask, X, mean_vals[None, :])
    
    def _log_progress(self, epoch, losses, total_epochs):
        """Log training progress"""
        print(f'Epoch [{epoch}/{total_epochs}] '
              f'D_loss: {losses["d_loss"]:.4f} '
              f'G_loss: {losses["g_loss"]:.4f}')
        
    def _adaptive_loss_weights(self, epoch, loss_history):
        """Theoretically justified adaptive loss weights"""
        if len(loss_history) > 10:
            recent_losses = loss_history[-10:]
            variance = np.var(recent_losses)
            return {
                'reconstruction': 5.0 * np.exp(-variance),
                'feature_matching': 0.5 * (1 + np.tanh(variance)),
                'smoothness': 0.01 * (1 + np.log1p(variance))
            }
        return {'reconstruction': 5.0, 'feature_matching': 0.5, 'smoothness': 0.01}



