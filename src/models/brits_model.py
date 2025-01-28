import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BRITSWrapper:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.device = device
        self.model = BRITS(input_dim=input_dim).to(self.device)
        self.trainer = ImputationTrainer(self.model)
    
    def _prepare_input(self, X, mask):
        """Prepare input data with gradient tracking"""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
        
        X = X.to(self.device).requires_grad_(True)
        mask = mask.to(self.device)
        
        # Handle NaN values
        X = torch.where(torch.isnan(X), torch.zeros_like(X), X)
        
        return X, mask
    
    def fit_transform(self, X, mask):
        try:
            # Prepare input data
            X, mask = self._prepare_input(X, mask)
            
            # Training parameters
            batch_size = min(32, X.size(0))
            n_epochs = 100
            
            # Training loop
            for epoch in range(n_epochs):
                total_loss = 0.0
                n_batches = 0
                
                for start_idx in range(0, X.size(0), batch_size):
                    end_idx = min(start_idx + batch_size, X.size(0))
                    
                    # Get batch data - use full data including CGR
                    batch_X = X[start_idx:end_idx]
                    batch_mask = mask[start_idx:end_idx]
                    
                    try:
                        # Training step
                        loss = self.trainer.train_step(batch_X, batch_mask)
                        total_loss += loss
                        n_batches += 1
                    except Exception as e:
                        print(f"Warning: Error in batch {start_idx}-{end_idx}: {str(e)}")
                        continue
                
                if epoch % 10 == 0 and n_batches > 0:
                    avg_loss = total_loss / n_batches
                    print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
            
            # Final imputation
            self.model.eval()
            imputed_data = []
            
            with torch.no_grad():
                for start_idx in range(0, X.size(0), batch_size):
                    end_idx = min(start_idx + batch_size, X.size(0))
                    batch_X = X[start_idx:end_idx]
                    batch_mask = mask[start_idx:end_idx]
                    
                    imputed = self.model(batch_X, batch_mask)
                    imputed = torch.where(batch_mask > 0.5, batch_X, imputed)
                    imputed_data.append(imputed)
            
            imputed = torch.cat(imputed_data, dim=0)
            
            return imputed.cpu().detach().numpy()
            
        except Exception as e:
            print(f"Error in BRITS: {str(e)}")
            return X.cpu().detach().numpy() if torch.is_tensor(X) else X


class BRITS(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim  # This should be the full dimension including CGR
        self.hidden_dim = hidden_dim
        
        self.forward_rnn = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        
        self.regression = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.input_dim)
        )
        
    def forward(self, x, mask):
        # Add sequence dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            mask = mask.unsqueeze(1)
            
        batch_size = x.size(0)
        
        # Initial hidden state
        h0 = torch.zeros(2, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(2, batch_size, self.hidden_dim).to(x.device)
        
        # Forward pass through LSTM
        out, _ = self.forward_rnn(x, (h0, c0))
        
        # Generate imputed values
        imputed = self.regression(out)
        
        # Remove sequence dimension if it was added
        if imputed.size(1) == 1:
            imputed = imputed.squeeze(1)
        
        return imputed


class ImputationTrainer:
    def __init__(self, model, learning_rate=1e-3, device=device):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.device = device
        
    def train_step(self, batch_X, batch_mask, indicating_mask=None):
        """Training step with proper gradient tracking"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Ensure tensors require gradients
        batch_X = batch_X.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = self.model(batch_X, batch_mask)
        
        if indicating_mask is not None:
            # Create holdout data
            X_holdout = batch_X.clone().detach()
            X_holdout = torch.where(indicating_mask.bool(), X_holdout, torch.zeros_like(X_holdout))
            
            # Compute loss on masked values
            loss = F.mse_loss(outputs * indicating_mask, X_holdout * indicating_mask)
        else:
            # Compute loss on all observed values
            loss = F.mse_loss(outputs * batch_mask, batch_X * batch_mask)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
