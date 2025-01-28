class DEGAINGenerator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        
        # Generator architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, 256),  # Double input_dim for concatenated noise
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, input_dim),
            nn.Tanh()  # Output activation
        )
        
    def forward(self, x, z):
        # Concatenate input with noise
        combined = torch.cat([x, z], dim=1)
        return self.net(combined)

class DEGAINDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        
        # Discriminator architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

class DEGAINImputer:
    def __init__(self, input_dim, n_epochs=100, inner_iterations=5, alpha_g=0.001, alpha_d=0.001, epsilon=1e-8):
        self.input_dim = input_dim
        self.device = device
        self.n_epochs = n_epochs
        self.inner_iterations = inner_iterations
        self.alpha_g = alpha_g
        self.alpha_d = alpha_d
        self.epsilon = epsilon
        
        # Initialize networks
        self.generator = DEGAINGenerator(input_dim).to(self.device)
        self.discriminator = DEGAINDiscriminator(input_dim).to(self.device)
        
        # Initialize scaler
        self.scaler = RobustScaler()
        
        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=alpha_g)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=alpha_d)
        
    def fit_transform(self, X, mask):
        try:
            # Handle NaN values first
            X_filled = X.copy()
            mean_vals = np.nanmean(X_filled, axis=0)
            for i in range(X.shape[1]):
                X_filled[np.isnan(X_filled[:, i]), i] = mean_vals[i]
            
            # Scale data
            X_scaled = self.scaler.fit_transform(X_filled)
            
            # Convert to tensors
            X = torch.FloatTensor(X_scaled).to(self.device)
            mask = torch.FloatTensor(mask).to(self.device)
            
            batch_size = min(128, X.shape[0])
            
            # Training loop
            for epoch in range(self.n_epochs):
                epoch_g_loss = 0
                epoch_d_loss = 0
                n_batches = 0
                
                for i in range(0, X.shape[0], batch_size):
                    end_idx = min(i + batch_size, X.shape[0])
                    batch_mask = mask[i:end_idx]
                    batch_X = X[i:end_idx]
                    curr_batch_size = end_idx - i
                    
                    for _ in range(self.inner_iterations):
                        # Train Discriminator
                        self.d_optimizer.zero_grad()
                        Z = torch.randn(curr_batch_size, self.input_dim).to(self.device)
                        fake_X = self.generator(batch_X, Z)
                        
                        real_pred = self.discriminator(batch_X)
                        fake_pred = self.discriminator(fake_X.detach())
                        
                        # Modified loss calculation with clipping
                        d_loss = -(torch.mean(torch.clamp(torch.log(real_pred + 1e-8), min=-100)) + 
                                 torch.mean(torch.clamp(torch.log(1 - fake_pred + 1e-8), min=-100)))
                        
                        if not torch.isnan(d_loss):
                            d_loss.backward()
                            self.d_optimizer.step()
                        
                        # Train Generator
                        self.g_optimizer.zero_grad()
                        Z = torch.randn(curr_batch_size, self.input_dim).to(self.device)
                        fake_X = self.generator(batch_X, Z)
                        fake_pred = self.discriminator(fake_X)
                        
                        mse_loss = F.mse_loss(fake_X * batch_mask, batch_X * batch_mask)
                        g_loss = -torch.mean(torch.clamp(torch.log(fake_pred + 1e-8), min=-100)) + mse_loss
                        
                        if not torch.isnan(g_loss):
                            g_loss.backward()
                            self.g_optimizer.step()
                        
                        if not torch.isnan(g_loss) and not torch.isnan(d_loss):
                            epoch_g_loss += g_loss.item()
                            epoch_d_loss += d_loss.item()
                    
                    n_batches += 1
                
                if n_batches > 0:
                    avg_g_loss = epoch_g_loss / (n_batches * self.inner_iterations)
                    avg_d_loss = epoch_d_loss / (n_batches * self.inner_iterations)
                    
                    if epoch % 10 == 0:
                        print(f'Epoch [{epoch}/{self.n_epochs}] G_loss: {avg_g_loss:.4f} D_loss: {avg_d_loss:.4f}')
            
            # Final imputation
            self.generator.eval()
            imputed_data = []
            
            with torch.no_grad():
                for i in range(0, X.shape[0], batch_size):
                    end_idx = min(i + batch_size, X.shape[0])
                    batch_X = X[i:end_idx]
                    batch_mask = mask[i:end_idx]
                    Z = torch.randn(end_idx - i, self.input_dim).to(self.device)
                    
                    fake_X = self.generator(batch_X, Z)
                    batch_imputed = torch.where(batch_mask.bool(), batch_X, fake_X)
                    imputed_data.append(batch_imputed)
            
            imputed = torch.cat(imputed_data, dim=0)
            return self.scaler.inverse_transform(imputed.cpu().numpy())
            
        except Exception as e:
            print(f"Error in DEGAIN: {str(e)}")
            # Return mean imputation as fallback
            X_filled = X.copy() if isinstance(X, np.ndarray) else X.cpu().numpy()
            mean_vals = np.nanmean(X_filled, axis=0)
            for i in range(X_filled.shape[1]):
                X_filled[~mask[:, i], i] = mean_vals[i]
            return X_filled