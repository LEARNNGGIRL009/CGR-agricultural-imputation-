import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams['font.family'] = 'Times New Roman'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#3. Ablation test result generation
# Base DLPIM Implementation
class EnhancedDLPIMGenerator(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Calculate input dimensions for each component
        self.feature_input_dim = input_dim - 1  # Excluding CGR column
        
        # Feature encoder with residual connections
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_input_dim + latent_dim, 256),
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
        
        # CGR processor with skip connections
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
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Decoder
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
            nn.Linear(128, self.feature_input_dim),  # Output only feature dimensions
            nn.Tanh()
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, noise, features, cgr):
        # Process features with noise
        x = torch.cat([features, noise], dim=1)
        encoded_features = self.feature_encoder(x)
        
        # Apply self attention
        encoded_features = encoded_features.unsqueeze(1)
        attended_features, _ = self.self_attention(
            encoded_features, 
            encoded_features, 
            encoded_features
        )
        attended_features = attended_features.squeeze(1)
        
        # Process CGR
        cgr_features = self.cgr_processor(cgr)
        
        # Combine features
        combined = torch.cat([attended_features, cgr_features], dim=1)
        
        # Generate output (features only)
        output = self.decoder(combined)
        
        # Concatenate with original CGR
        return output

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
        return x + 0.1 * self.block(x)

# Ablation Study Variants
class DLPIMWithoutCGR(EnhancedDLPIMGenerator):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__(input_dim, latent_dim)
        # Replace CGR processor with zero tensor
        self.cgr_removed = True
        
    def forward(self, noise, features, cgr):
        x = torch.cat([features, noise], dim=1)
        encoded_features = self.feature_encoder(x)
        
        # Apply attention
        encoded_features = encoded_features.unsqueeze(1)
        attended_features, _ = self.self_attention(
            encoded_features, 
            encoded_features, 
            encoded_features
        )
        attended_features = attended_features.squeeze(1)
        
        # Create zero tensor for CGR features
        batch_size = features.shape[0]
        dummy_cgr_features = torch.zeros(batch_size, 32).to(features.device)
        
        # Combine features with dummy CGR features
        combined = torch.cat([attended_features, dummy_cgr_features], dim=1)
        
        # Generate output
        output = self.decoder(combined)
        return output

class DLPIMWithoutAttention(EnhancedDLPIMGenerator):
    def forward(self, noise, features, cgr):
        x = torch.cat([features, noise], dim=1)
        encoded_features = self.feature_encoder(x)
        # Process CGR
        cgr_features = self.cgr_processor(cgr)
        
        # Skip attention, use encoded features directly
        combined = torch.cat([encoded_features, cgr_features], dim=1)
        output = self.decoder(combined)
        return output

class DLPIMSingleHead(EnhancedDLPIMGenerator):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__(input_dim, latent_dim)
        # Replace multi-head attention with single-head
        self.self_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=1,  # Single head
            dropout=0.1,
            batch_first=True
        )

class DLPIMWithoutResidual(EnhancedDLPIMGenerator):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__(input_dim, latent_dim)
        # Replace residual blocks with standard layers
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_input_dim + latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(128)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(128 + 32, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, self.feature_input_dim),
            nn.Tanh()
        )

class DLPIMMinimal(EnhancedDLPIMGenerator):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__(input_dim, latent_dim)
        # Minimal architecture without special components
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_input_dim + latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.feature_input_dim)
        )
        
    def forward(self, noise, features, cgr):
        x = torch.cat([features, noise], dim=1)
        return self.encoder(x)
        
    def forward(self, noise, features, cgr):
        x = torch.cat([features, noise], dim=1)
        return self.encoder(x)

class AblationDLPIMImputer:
    def __init__(self, input_dim, variant_type="full", latent_dim=64):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.variant_type = variant_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks based on variant
        self.generator = self._create_generator()
        self.scaler = RobustScaler()
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=1e-4,
            betas=(0.5, 0.999),
            weight_decay=1e-5
        )

    def _create_generator(self):
        variants = {
            "no_cgr": DLPIMWithoutCGR,
            "no_attention": DLPIMWithoutAttention,
            "single_head": DLPIMSingleHead,
            "no_residual": DLPIMWithoutResidual,
            "minimal": DLPIMMinimal,
            "full": EnhancedDLPIMGenerator
        }
        
        generator_class = variants.get(self.variant_type, EnhancedDLPIMGenerator)
        return generator_class(self.input_dim, self.latent_dim).to(self.device)

    def fit_transform(self, X, mask):
        """Fit the model and transform input data"""
        X_tensor = torch.FloatTensor(self.scaler.fit_transform(X)).to(self.device)
        mask_tensor = torch.FloatTensor(mask).to(self.device)
        
        # Split features and CGR
        features = X_tensor[:, :-1]  # All columns except the last one
        cgr = X_tensor[:, -1:].detach()  # Last column
        feature_mask = mask_tensor[:, :-1]  # Mask for features
        
        self.generator.train()
        for epoch in range(100):  # Simplified training loop
            noise = torch.randn(X.shape[0], self.latent_dim).to(self.device)
            
            # Generate imputed values for features only
            imputed_features = self.generator(noise, features, cgr)
            
            # Compute loss on features only
            loss = F.mse_loss(
                imputed_features * feature_mask, 
                features * feature_mask
            )
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Generate final imputation
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(X.shape[0], self.latent_dim).to(self.device)
            imputed_features = self.generator(noise, features, cgr)
            
            # Combine imputed features with original CGR
            imputed = torch.cat([imputed_features, cgr], dim=1)
            
        return self.scaler.inverse_transform(imputed.cpu().numpy())

def run_ablation_study(data_path, missing_rates=[0.1, 0.2, 0.4, 0.6]):
    """Run ablation study on DLPIM variants with multiple missing rates"""
    # Load data
    data = pd.read_csv(data_path)
    X = data.values
    
    variants = ["DLPIM", "No_CGR", "No_Attention", "Single_Head", "No_Rec_Loss", "No_Feat_Loss", "No_TV_Loss"]
    results = []
    
    for variant in variants:
        for rate in missing_rates:
            print(f"\nTesting {variant} variant with {rate*100}% missing rate...")
            
            # Create missing pattern
            mask = np.random.rand(*X.shape) > rate
            mask[:, -1] = True  # Keep CGR column intact
            
            # Initialize and run model
            imputer = AblationDLPIMImputer(X.shape[1], variant_type=variant.lower())
            
            start_time = time.time()
            imputed = imputer.fit_transform(X, mask)
            runtime = time.time() - start_time
            
            # Calculate metrics
            mse = mean_squared_error(X[~mask], imputed[~mask])
            mae = mean_absolute_error(X[~mask], imputed[~mask])
            r2 = r2_score(X[~mask], imputed[~mask])
            rmse = np.sqrt(mse)
            
            results.append({
                'Model Variant': variant,
                'Missing Rate (%)': f"{rate*100}%",
                'MSE': mse,
                'MAE': mae,
                'R²': r2,
                'RMSE': rmse
            })
            
            print(f"MSE: {mse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R²: {r2:.4f}")
            print(f"RMSE: {rmse:.4f}")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run ablation study
    results = run_ablation_study('data.csv')
    
    # Create results DataFrame
    df_results = pd.DataFrame(results).T
    print("\nAblation Study Results:")
    print(df_results)
    
    # Save results
    df_results.to_csv('ablation_results.csv')

#4. Ablation result plots

plt.rcParams['font.family'] = 'Times New Roman'

def create_ablation_comparison_combined(file_path):
    """
    Create a single bar chart comparing performance across different model components
    with Times New Roman font
    """
    # Read CSV file
    results_df = pd.read_csv(file_path, encoding='cp1252')
    
    # Sort the model variants in the desired order
    model_order = ['DLPIM', 'No_CGR', 'No_Attention', 'Single_Head', 'No_Rec_Loss', 'No_Feat_Loss', 'No_TV_Loss']
    results_df['Model Variant'] = pd.Categorical(results_df['Model Variant'], categories=model_order, ordered=True)
    results_df = results_df.sort_values(['Model Variant', 'Missing Rate'])
    
    # Get unique model variants and missing rates
    model_variants = model_order
    missing_rates = sorted(results_df['Missing Rate'].unique())
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Set width of bars and positions of the bars
    bar_width = 0.21
    r = np.arange(len(model_variants))
    
    # Define colors for different missing rates
    colors = ['#2196F3', '#4CAF50', '#FFC107', '#F44336']
    
    # Create bars for each missing rate
    bars = []
    for idx, rate in enumerate(missing_rates):
        rate_data = results_df[results_df['Missing Rate'] == rate].copy()
        rate_data = rate_data.sort_values('Model Variant')
        
        # Round values for consistent comparison
        rate_data['R²'] = rate_data['R²'].round(5)
        
        # Create bars for this missing rate
        current_bars = ax.bar(r + idx * bar_width, 
                            rate_data['R²'],
                            bar_width,
                            label=f'Missing Rate {rate}',
                            color=colors[idx])
        bars.append(current_bars)
        
        # Add value labels on top of bars
        for i, bar in enumerate(current_bars):
            height = bar.get_height()
            
            # Calculate vertical offset for overlapping values
            vertical_offset = 0.001
            if idx > 0:
                prev_heights = [prev_bars[i].get_height() for prev_bars in bars[:idx]]
                if any(abs(height - prev_height) < 1e-5 for prev_height in prev_heights):
                    vertical_offset = 0.002 * idx
            
            # Only add label if height is not zero
            if height > 0:
                ax.text(bar.get_x() + bar_width/2,
                       height + vertical_offset,
                       f'{height:.3f}',
                       ha='center',
                       va='bottom',
                       fontsize=10,
                       family='Times New Roman')
    
    # Customize the plot with Times New Roman font
    #ax.set_title('Ablation Study: Impact of Different Components', pad=20, fontsize=10, family='Times New Roman')
    ax.set_xlabel('Model Components', 
                 labelpad=10, fontsize=10, family='Times New Roman')
    ax.set_ylabel('Performance Score (R²)', 
                 labelpad=10, fontsize=10, family='Times New Roman')
    
    # Set x-axis ticks with Times New Roman font
    ax.set_xticks(r + bar_width * (len(missing_rates)-1)/2)
    ax.set_xticklabels(model_variants, 
                       rotation=45, ha='right', 
                       fontsize=10, family='Times New Roman')
    
    # Set y-axis ticks with Times New Roman font
    ax.set_ylim(0.84, 1.0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend with Times New Roman font
    legend = ax.legend(title='Missing Rate',
                      bbox_to_anchor=(1.02, 1),
                      loc='upper right',
                      borderaxespad=0,
                      prop={'family': 'Times New Roman', 'size': 10})
    legend.get_title().set_family('Times New Roman')
    legend.get_title().set_fontsize(10)
    
    plt.tight_layout()
    
    return fig

if __name__ == "__main__":
    # Generate visualization
    fig = create_ablation_comparison_combined('ablation_results.csv')
    
    # Save figure
    fig.savefig('ablation_component_comparison.tiff', 
                dpi=600, 
                bbox_inches='tight',
                pil_kwargs={'compression': 'tiff_lzw'})
