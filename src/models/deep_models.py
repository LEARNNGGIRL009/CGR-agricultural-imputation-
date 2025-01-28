import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_models import DeepImputationBase
from ..utils.data_processor import create_missing_pattern

class EnhancedGAIN(DeepImputationBase):
    def __init__(self, data_dim):
        super().__init__()
        self.data_dim = data_dim
        self.device = device
        
        # Generator - uses the full input dimension
        self.generator = nn.Sequential(
            nn.Linear(self.data_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.data_dim)
        ).to(self.device)
        
        # Discriminator - uses the full input dimension
        self.discriminator = nn.Sequential(
            nn.Linear(self.data_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = torch.optim.Adam(self.generator.parameters())
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters())

    def _prepare_input(self, X, mask):
        """
        Prepare input data for GAIN model.
        """
        # Convert to numpy if tensors
        if torch.is_tensor(X):
            X = X.cpu().numpy()
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
            
        # Handle NaN values with mean imputation
        X_processed = X.copy()
        means = np.nanmean(X_processed, axis=0)
        for i in range(X.shape[1]):
            X_processed[np.isnan(X_processed[:, i]), i] = means[i]
            
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_processed).to(self.device)
        mask_tensor = torch.FloatTensor(mask).to(self.device)
        mask_bool = mask_tensor.bool()
        
        return X_tensor, mask_tensor, mask_bool
        
    def fit_transform(self, X, mask):
        try:
            # Prepare input data
            X, mask, mask_bool = self._prepare_input(X, mask)
            
            # Training parameters
            batch_size = min(128, X.size(0))
            n_epochs = 100
            
            optimizer_G = torch.optim.Adam(self.generator.parameters())
            optimizer_D = torch.optim.Adam(self.discriminator.parameters())
            
            # Training loop
            for epoch in range(n_epochs):
                total_g_loss = 0
                total_d_loss = 0
                n_batches = 0
                
                for i in range(0, X.size(0), batch_size):
                    batch_end = min(i + batch_size, X.size(0))
                    batch_X = X[i:batch_end]
                    batch_mask = mask_bool[i:batch_end]
                    
                    # Train Discriminator
                    optimizer_D.zero_grad()
                    noise = torch.randn(batch_end - i, self.data_dim).to(self.device)
                    fake_data = self.generator(noise)
                    
                    d_loss = -torch.mean(
                        torch.log(self.discriminator(batch_X) + 1e-8) + 
                        torch.log(1 - self.discriminator(fake_data.detach()) + 1e-8)
                    )
                    
                    d_loss.backward()
                    optimizer_D.step()
                    
                    # Train Generator
                    optimizer_G.zero_grad()
                    fake_data = self.generator(noise)
                    g_loss = -torch.mean(torch.log(self.discriminator(fake_data) + 1e-8))
                    
                    g_loss.backward()
                    optimizer_G.step()
                    
                    total_g_loss += g_loss.item()
                    total_d_loss += d_loss.item()
                    n_batches += 1
                    
                if epoch % 10 == 0:
                    print(f'Epoch [{epoch}/100] G_loss: {total_g_loss/n_batches:.4f} D_loss: {total_d_loss/n_batches:.4f}')
            
            # Final imputation
            with torch.no_grad():
                imputed_data = []
                for i in range(0, X.size(0), batch_size):
                    batch_end = min(i + batch_size, X.size(0))
                    batch_X = X[i:batch_end]
                    batch_mask = mask_bool[i:batch_end]
                    
                    noise = torch.randn(batch_end - i, self.data_dim).to(self.device)
                    fake = self.generator(noise)
                    imputed = torch.where(batch_mask, batch_X, fake)
                    imputed_data.append(imputed)
                
                imputed = torch.cat(imputed_data, dim=0)
                
            return imputed.cpu().numpy()
            
        except Exception as e:
            print(f"Error in GAIN: {str(e)}")
            if np.isnan(X).any():
                print("Error in GAIN: Input contains NaN.")
            if torch.is_tensor(X):
                print("Debug info for GAIN:")
                print(f"Input shapes - X: {X.shape}, mask: {mask.shape}")
            else:
                print("Debug info for GAIN:")
                print(f"Input shapes - X: {X.shape}, mask: {mask.shape}")
            # Return mean imputation as fallback
            X_filled = X.cpu().numpy() if torch.is_tensor(X) else X.copy()
            mean_vals = np.nanmean(X_filled, axis=0)
            for i in range(X_filled.shape[1]):
                X_filled[~mask[:, i], i] = mean_vals[i]
            return X_filled