from imputation_analysis import EnhancedDLPIMImputer, BRITSWrapper


"""
Integrated script: Imputation quality → Yield prediction impact
Add this section to your existing imputation_analysis.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Use existing device setup from your main code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class YieldPredictor(nn.Module):
    """Neural network for yield prediction"""
    def __init__(self, input_dim=6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.network(x)


def create_synthetic_yield(data):
    """Create synthetic yield from nutrient data"""
    Na = data[:, 1]
    K = data[:, 2]
    Mg = data[:, 3]
    Ca = data[:, 4]
    CGR = data[:, 5]
    
    yield_values = (
        0.4 * K +
        0.3 * CGR * 10 +
        0.2 * Ca +
        0.1 * Mg
    )
    
    np.random.seed(42)
    noise = np.random.normal(0, 0.05 * np.std(yield_values), len(yield_values))
    yield_values = yield_values + noise
    
    yield_min, yield_max = yield_values.min(), yield_values.max()
    yield_values = 2 + 6 * (yield_values - yield_min) / (yield_max - yield_min)
    
    return yield_values


def train_yield_model(X_train, y_train, X_val, y_val, epochs=150, verbose=False):
    """Train yield prediction model"""
    model = YieldPredictor(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs.squeeze(), y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs.squeeze(), y_val_t)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 30 == 0:
            print(f'  Epoch [{epoch+1}/{epochs}] Train: {loss.item():.4f}, Val: {val_loss.item():.4f}')
        
        if patience_counter >= patience:
            if verbose:
                print(f'  Early stopping at epoch {epoch+1}')
            break
    
    model.load_state_dict(best_model_state)
    return model


def evaluate_yield_prediction(model, X_test, y_test):
    """Evaluate model on test set"""
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        y_pred = model(X_test_t).cpu().numpy().squeeze()
    
    return {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'predictions': y_pred
    }


def run_yield_experiment_with_imputation(data_path, missing_rates=[0.1, 0.2, 0.4, 0.6]):
    """
    Main experiment: Compare yield prediction quality across imputation methods
    """
    print("="*70)
    print("YIELD PREDICTION WITH IMPUTED DATA EXPERIMENT")
    print("="*70)
    
    # Load data
    df = pd.read_csv(data_path)
    data = df.values
    print(f"\nDataset: {data.shape[0]} samples, {data.shape[1]} features")
    
    # Create synthetic yield
    yield_values = create_synthetic_yield(data)
    print(f"Synthetic yield: {yield_values.mean():.2f} ± {yield_values.std():.2f} kg/plant")
    
    # Create indices for consistent splitting
    np.random.seed(42)
    n_samples = len(data)
    indices = np.random.permutation(n_samples)
    
    # Split indices
    test_size = int(0.2 * n_samples)
    val_size = int(0.15 * (n_samples - test_size))
    
    test_idx = indices[:test_size]
    val_idx = indices[test_size:test_size + val_size]
    train_idx = indices[test_size + val_size:]
    
    print(f"\nSplits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # Store all results
    all_results = {}
    
    # Imputation methods to test
    from sklearn.impute import KNNImputer, IterativeImputer
    
    # Import your deep learning imputers
    # Make sure these classes are available in your environment
    methods = {
        'Mean': 'mean',  # Simple baseline
        'KNN': KNNImputer(n_neighbors=5),
        'MICE': IterativeImputer(max_iter=10, random_state=42),
        'DLPIM': EnhancedDLPIMImputer(data.shape[1]-1),  # Exclude CGR
        'BRITS': BRITSWrapper(input_dim=data.shape[1])
    }
    
    # For each missing rate
    for rate in missing_rates:
        print(f"\n{'='*70}")
        print(f"MISSING RATE: {int(rate*100)}%")
        print(f"{'='*70}")
        
        rate_results = {}
        
        # 1. BASELINE: Complete data (no missing values)
        print(f"\n[1/4] Training on COMPLETE data (baseline)...")
        
        # Get splits from complete data
        X_train_complete = data[train_idx]
        X_val_complete = data[val_idx]
        X_test_complete = data[test_idx]
        y_train = yield_values[train_idx]
        y_val = yield_values[val_idx]
        y_test = yield_values[test_idx]
        
        # Standardize
        scaler_complete = StandardScaler()
        X_train_scaled = scaler_complete.fit_transform(X_train_complete)
        X_val_scaled = scaler_complete.transform(X_val_complete)
        X_test_scaled = scaler_complete.transform(X_test_complete)
        
        # Train
        model_complete = train_yield_model(X_train_scaled, y_train, X_val_scaled, y_val)
        results_complete = evaluate_yield_prediction(model_complete, X_test_scaled, y_test)
        
        rate_results['Complete'] = results_complete
        print(f"  Complete → MAE: {results_complete['MAE']:.4f}, R²: {results_complete['R2']:.4f}")
        
        # 2. Create missing data pattern
        from sklearn.impute import SimpleImputer
        
        # Create mask (CGR always observed)
        mask = np.ones_like(data, dtype=bool)
        for col in range(data.shape[1] - 1):  # Exclude CGR
            missing_idx = np.random.choice(n_samples, size=int(n_samples * rate), replace=False)
            mask[missing_idx, col] = False
        mask[:, -1] = True  # CGR always observed
        
        # Apply mask
        data_missing = data.copy()
        data_missing[~mask] = np.nan
        
        # 3. Test each imputation method
        for i, (method_name, imputer) in enumerate(methods.items(), start=2):
            print(f"\n[{i}/6] Training on {method_name}-imputed data...")
            
            try:
                # Impute
                if method_name == 'Mean':
                    imputer_obj = SimpleImputer(strategy='mean')
                    data_imputed = imputer_obj.fit_transform(data_missing)
                elif method_name in ['DLPIM', 'BRITS']:
                    # Deep learning methods need standardization
                    print(f"  Standardizing data for {method_name}...")
                    mean_vals = np.nanmean(data_missing, axis=0)
                    std_vals = np.nanstd(data_missing, axis=0) + 1e-8
                    data_standardized = (data_missing - mean_vals) / std_vals
                    
                    print(f"  Running {method_name} imputation...")
                    data_imputed_std = imputer.fit_transform(data_standardized, mask)
                    
                    # Inverse standardization
                    data_imputed = data_imputed_std * std_vals + mean_vals
                    print(f"  {method_name} imputation completed")
                else:
                    data_imputed = imputer.fit_transform(data_missing)
                
                # Get splits from imputed data
                X_train_imp = data_imputed[train_idx]
                X_val_imp = data_imputed[val_idx]
                X_test_imp = data_imputed[test_idx]
                
                # Standardize for model training
                scaler_imp = StandardScaler()
                X_train_imp_scaled = scaler_imp.fit_transform(X_train_imp)
                X_val_imp_scaled = scaler_imp.transform(X_val_imp)
                X_test_imp_scaled = scaler_imp.transform(X_test_imp)
                
                # Train
                print(f"  Training yield prediction model...")
                model_imp = train_yield_model(X_train_imp_scaled, y_train, X_val_imp_scaled, y_val)
                results_imp = evaluate_yield_prediction(model_imp, X_test_imp_scaled, y_test)
                
                # Calculate degradation
                mae_degradation = ((results_imp['MAE'] - results_complete['MAE']) / 
                                  results_complete['MAE']) * 100
                r2_degradation = ((results_complete['R2'] - results_imp['R2']) / 
                                 results_complete['R2']) * 100
                
                results_imp['MAE_degradation_%'] = mae_degradation
                results_imp['R2_degradation_%'] = r2_degradation
                
                rate_results[method_name] = results_imp
                
                print(f"  {method_name} → MAE: {results_imp['MAE']:.4f} (+{mae_degradation:.1f}%), "
                      f"R²: {results_imp['R2']:.4f} (-{r2_degradation:.1f}%)")
                
            except Exception as e:
                print(f"  ERROR with {method_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
            
            # Clean up GPU memory after deep learning methods
            if method_name in ['DLPIM', 'BRITS'] and torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"  GPU memory cleared")
        
        all_results[f'{int(rate*100)}%'] = rate_results
        
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save and visualize results
    save_and_plot_yield_results(all_results)
    
    return all_results


def save_and_plot_yield_results(results):
    """Save results and create visualizations"""
    
    # Prepare data for plotting
    plot_data = []
    for missing_rate, methods in results.items():
        for method, metrics in methods.items():
            plot_data.append({
                'Missing_Rate': missing_rate,
                'Method': method,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'R2': metrics['R2'],
                'MAE_Degradation': metrics.get('MAE_degradation_%', 0),
                'R2_Degradation': metrics.get('R2_degradation_%', 0)
            })
    
    df = pd.DataFrame(plot_data)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. MAE comparison
    ax1 = axes[0, 0]
    sns.barplot(data=df, x='Missing_Rate', y='MAE', hue='Method', ax=ax1)
    ax1.set_title('Mean Absolute Error by Missing Rate', fontsize=14, fontweight='bold')
    ax1.set_ylabel('MAE (kg/plant)')
    ax1.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. R² comparison
    ax2 = axes[0, 1]
    sns.barplot(data=df, x='Missing_Rate', y='R2', hue='Method', ax=ax2)
    ax2.set_title('R² Score by Missing Rate', fontsize=14, fontweight='bold')
    ax2.set_ylabel('R² Score')
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Target')
    ax2.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. MAE Degradation
    ax3 = axes[1, 0]
    df_deg = df[df['Method'] != 'Complete']
    sns.barplot(data=df_deg, x='Missing_Rate', y='MAE_Degradation', hue='Method', ax=ax3)
    ax3.set_title('MAE Degradation vs Complete Data', fontsize=14, fontweight='bold')
    ax3.set_ylabel('MAE Increase (%)')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. R² Degradation
    ax4 = axes[1, 1]
    sns.barplot(data=df_deg, x='Missing_Rate', y='R2_Degradation', hue='Method', ax=ax4)
    ax4.set_title('R² Degradation vs Complete Data', fontsize=14, fontweight='bold')
    ax4.set_ylabel('R² Decrease (%)')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('yield_imputation_comparison.pdf', dpi=300, bbox_inches='tight')
    print("\n✓ Plots saved to: yield_imputation_comparison.pdf")
    plt.show()
    
    # Save to Excel
    df.to_excel('yield_imputation_results.xlsx', index=False)
    print("✓ Results saved to: yield_imputation_results.xlsx")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: Best Performing Methods")
    print("="*70)
    
    for rate in df['Missing_Rate'].unique():
        rate_data = df[df['Missing_Rate'] == rate]
        rate_data_no_complete = rate_data[rate_data['Method'] != 'Complete']
        
        if len(rate_data_no_complete) > 0:
            best_mae = rate_data_no_complete.loc[rate_data_no_complete['MAE'].idxmin()]
            best_r2 = rate_data_no_complete.loc[rate_data_no_complete['R2'].idxmax()]
            
            print(f"\nMissing Rate: {rate}")
            print(f"  Best MAE: {best_mae['Method']} ({best_mae['MAE']:.4f})")
            print(f"  Best R²:  {best_r2['Method']} ({best_r2['R2']:.4f})")


# Add this to your main() function
def main_integrated():
    """Run both imputation comparison AND yield prediction"""
    
    print("STEP 1: Basic imputation comparison")
    print("="*70)
    # Your existing run_extended_comparison() here if needed
    
    print("\n\nSTEP 2: Yield prediction with imputed data")
    print("="*70)
    results = run_yield_experiment_with_imputation(
        'Nutrients_No_missing_arranged_in_pattern.csv',
        missing_rates=[0.1, 0.2, 0.4, 0.6]
    )
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - yield_imputation_comparison.pdf")
    print("  - yield_imputation_results.xlsx")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main_integrated()
    