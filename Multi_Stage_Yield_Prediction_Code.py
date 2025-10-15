"""
Multi-Stage Yield Prediction - Simplified Version
Uses existing missing pattern generation from Missing_pattern_generation_sensor_alike.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import KNNImputer, IterativeImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Import your existing missing pattern function
from Missing_pattern_generation_sensor_alike import create_realistic_missing_pattern
from imputation_analysis import EnhancedDLPIMImputer, BRITSWrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# SIMPLE 3-STAGE NEURAL NETWORKS
# ============================================================================

class StagePredictor(nn.Module):
    """Generic predictor for all 3 stages"""
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)


def train_model(model, X_train, y_train, X_val, y_val, epochs=100):
    """Quick training function"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    best_loss = float('inf')
    patience, counter = 15, 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train_t).squeeze(), y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t).squeeze(), y_val_t)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    
    model.load_state_dict(best_state)
    return model


def predict(model, X):
    """Get predictions"""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        return model(X_t).cpu().numpy().squeeze()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_multistage_prediction(csv_path='Nutrients_No_missing_arranged_in_pattern.csv'):
    """
    Complete 3-stage prediction pipeline
    """
    
    print("="*80)
    print("MULTI-STAGE YIELD PREDICTION")
    print("="*80)
    
    # Load data
    df = pd.read_csv(csv_path)
    data = df.values
    print(f"\nDataset: {data.shape[0]} samples × {data.shape[1]} features")
    
    # Calculate cumulative CGR as target (proxy for yield)
    cgr_col = 5  # CGR is column index 5
    cumulative_cgr = np.cumsum(data[:, cgr_col])
    print(f"Target: Cumulative CGR (range: {cumulative_cgr.min():.2f} - {cumulative_cgr.max():.2f})")
    
    # Split data into train/val/test
    indices = np.arange(len(data))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    print(f"\nSplits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # Store results
    all_results = {}
    
    # ========================================================================
    # BASELINE: Complete Data
    # ========================================================================
    
    print("\n" + "="*80)
    print("BASELINE: Training on COMPLETE DATA")
    print("="*80)
    
    baseline = train_three_stages(
        data, cumulative_cgr, train_idx, val_idx, test_idx, 
        method_name="Complete"
    )
    all_results['Complete'] = baseline
    
    # ========================================================================
    # TEST WITH MISSING DATA + IMPUTATION
    # ========================================================================
    
    missing_rates = [0.1, 0.2, 0.4, 0.6]
    
    for rate in missing_rates:
        print(f"\n{'='*80}")
        print(f"MISSING RATE: {int(rate*100)}%")
        print(f"{'='*80}")
        
        # Generate missing pattern using YOUR existing function
        print(f"\nGenerating realistic missing pattern ({int(rate*100)}%)...")
        data_missing, mask = create_realistic_missing_pattern(
            data.copy(), 
            rate=rate,
            temporal_gradient=True,
            sensor_reliability=True,
            block_missing=True,
            random_seed=42
        )
        
        rate_results = {}
        
        # Test each imputation method
        methods = {
            'KNN': KNNImputer(n_neighbors=5),
            'MICE': IterativeImputer(max_iter=10, random_state=42),
            'DLPIM': EnhancedDLPIMImputer(data.shape[1] - 1),
            'BRITS': BRITSWrapper(input_dim=data.shape[1])
        }
        
        for method_name, imputer in methods.items():
            print(f"\n[{method_name}] Imputing and training...")
            
            try:
                # Impute missing data
                if method_name in ['DLPIM', 'BRITS']:
                    data_imputed = impute_deep_learning(data_missing, mask, imputer)
                else:
                    data_imputed = imputer.fit_transform(data_missing)
                
                # Train 3 stages
                results = train_three_stages(
                    data_imputed, cumulative_cgr, train_idx, val_idx, test_idx,
                    method_name=method_name
                )
                
                # Calculate degradation vs baseline
                for stage in ['Stage1', 'Stage2', 'Stage3']:
                    mae_deg = ((results[stage]['MAE'] - baseline[stage]['MAE']) / 
                              baseline[stage]['MAE']) * 100
                    r2_deg = ((baseline[stage]['R2'] - results[stage]['R2']) / 
                             baseline[stage]['R2']) * 100
                    results[stage]['MAE_deg_%'] = mae_deg
                    results[stage]['R2_deg_%'] = r2_deg
                
                rate_results[method_name] = results
                
                # Print Stage 3 (final) results
                s3 = results['Stage3']
                print(f"  Stage 3: MAE={s3['MAE']:.4f} (+{s3['MAE_deg_%']:.1f}%), "
                      f"R²={s3['R2']:.4f} (-{s3['R2_deg_%']:.1f}%)")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
        
        all_results[f'{int(rate*100)}%'] = rate_results
    
    # Save and visualize
    save_results_table(all_results)
    create_plots(all_results)
    
    return all_results


def train_three_stages(data, cumulative_cgr, train_idx, val_idx, test_idx, method_name):
    """
    Train all 3 stages sequentially
    
    Stage 1: [EC, Na, K, Mg, Ca] → Cumulative CGR (early)
    Stage 2: [EC, Na, K, Mg, Ca, Stage1_pred] → Cumulative CGR (mid)
    Stage 3: [EC, Na, K, Mg, Ca, Stage1_pred, Stage2_pred] → Cumulative CGR (final)
    """
    
    results = {}
    
    # Base features (columns 0-4: EC, Na, K, Mg, Ca)
    base_features = data[:, :5]
    target = cumulative_cgr
    
    # ========================================================================
    # STAGE 1: Early Growth Prediction
    # ========================================================================
    
    print(f"  [{method_name}] Stage 1 training...")
    
    # Features: EC, Na, K, Mg, Ca (5 features)
    X1 = base_features
    y1 = target
    
    # Split
    X1_train = X1[train_idx]
    X1_val = X1[val_idx]
    X1_test = X1[test_idx]
    y1_train = y1[train_idx]
    y1_val = y1[val_idx]
    y1_test = y1[test_idx]
    
    # Standardize
    scaler1 = StandardScaler()
    X1_train_s = scaler1.fit_transform(X1_train)
    X1_val_s = scaler1.transform(X1_val)
    X1_test_s = scaler1.transform(X1_test)
    
    # Train
    model1 = StagePredictor(input_dim=5).to(device)
    model1 = train_model(model1, X1_train_s, y1_train, X1_val_s, y1_val)
    
    # Predict
    y1_pred = predict(model1, X1_test_s)
    
    # Get predictions for ALL data (needed for Stage 2)
    X1_all_s = scaler1.transform(X1)
    stage1_preds_all = predict(model1, X1_all_s)
    
    # Metrics
    results['Stage1'] = {
        'MAE': mean_absolute_error(y1_test, y1_pred),
        'RMSE': np.sqrt(mean_squared_error(y1_test, y1_pred)),
        'R2': r2_score(y1_test, y1_pred)
    }
    
    # ========================================================================
    # STAGE 2: Mid-Season Prediction
    # ========================================================================
    
    print(f"  [{method_name}] Stage 2 training...")
    
    # Features: EC, Na, K, Mg, Ca + Stage1 prediction (6 features)
    X2 = np.column_stack([base_features, stage1_preds_all])
    y2 = target
    
    # Split
    X2_train = X2[train_idx]
    X2_val = X2[val_idx]
    X2_test = X2[test_idx]
    y2_train = y2[train_idx]
    y2_val = y2[val_idx]
    y2_test = y2[test_idx]
    
    # Standardize
    scaler2 = StandardScaler()
    X2_train_s = scaler2.fit_transform(X2_train)
    X2_val_s = scaler2.transform(X2_val)
    X2_test_s = scaler2.transform(X2_test)
    
    # Train
    model2 = StagePredictor(input_dim=6).to(device)
    model2 = train_model(model2, X2_train_s, y2_train, X2_val_s, y2_val)
    
    # Predict
    y2_pred = predict(model2, X2_test_s)
    
    # Get predictions for ALL data (needed for Stage 3)
    X2_all_s = scaler2.transform(X2)
    stage2_preds_all = predict(model2, X2_all_s)
    
    # Metrics
    results['Stage2'] = {
        'MAE': mean_absolute_error(y2_test, y2_pred),
        'RMSE': np.sqrt(mean_squared_error(y2_test, y2_pred)),
        'R2': r2_score(y2_test, y2_pred)
    }
    
    # ========================================================================
    # STAGE 3: FINAL Yield Prediction
    # ========================================================================
    
    print(f"  [{method_name}] Stage 3 training (FINAL)...")
    
    # Features: EC, Na, K, Mg, Ca + Stage1 + Stage2 (7 features)
    X3 = np.column_stack([base_features, stage1_preds_all, stage2_preds_all])
    y3 = target
    
    # Split
    X3_train = X3[train_idx]
    X3_val = X3[val_idx]
    X3_test = X3[test_idx]
    y3_train = y3[train_idx]
    y3_val = y3[val_idx]
    y3_test = y3[test_idx]
    
    # Standardize
    scaler3 = StandardScaler()
    X3_train_s = scaler3.fit_transform(X3_train)
    X3_val_s = scaler3.transform(X3_val)
    X3_test_s = scaler3.transform(X3_test)
    
    # Train
    model3 = StagePredictor(input_dim=7).to(device)
    model3 = train_model(model3, X3_train_s, y3_train, X3_val_s, y3_val)
    
    # Predict
    y3_pred = predict(model3, X3_test_s)
    
    # Metrics
    results['Stage3'] = {
        'MAE': mean_absolute_error(y3_test, y3_pred),
        'RMSE': np.sqrt(mean_squared_error(y3_test, y3_pred)),
        'R2': r2_score(y3_test, y3_pred)
    }
    
    print(f"  [{method_name}] Complete: Stage3 MAE={results['Stage3']['MAE']:.4f}, "
          f"R²={results['Stage3']['R2']:.4f}")
    
    return results


def impute_deep_learning(data_missing, mask, imputer):
    """Impute using DLPIM or BRITS"""
    # Standardize
    mean_vals = np.nanmean(data_missing, axis=0)
    std_vals = np.nanstd(data_missing, axis=0) + 1e-8
    data_std = (data_missing - mean_vals) / std_vals
    
    # Impute
    data_imputed_std = imputer.fit_transform(data_std, mask)
    
    # Inverse transform
    data_imputed = data_imputed_std * std_vals + mean_vals
    
    return data_imputed


def save_results_table(all_results):
    """Save to Excel - Table 4.8.1 format"""
    
    rows = []
    for missing_rate, methods in all_results.items():
        for method, stages in methods.items():
            for stage_name, metrics in stages.items():
                rows.append({
                    'Missing_Rate': missing_rate,
                    'Method': method,
                    'Stage': stage_name,
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'R2': metrics['R2'],
                    'MAE_Degradation_%': metrics.get('MAE_deg_%', 0),
                    'R2_Degradation_%': metrics.get('R2_deg_%', 0)
                })
    
    df = pd.DataFrame(rows)
    df.to_excel('Table_4.8.1_MultiStage_Results.xlsx', index=False)
    print("\n✓ Table 4.8.1 saved: Table_4.8.1_MultiStage_Results.xlsx")


def create_plots(all_results):
    """Create Figure 4.8.1 - Performance comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Multi-Stage Yield Prediction Performance', fontsize=16, fontweight='bold')
    
    # Prepare data
    plot_data = []
    for missing_rate, methods in all_results.items():
        if missing_rate == 'Complete':
            continue
        for method, stages in methods.items():
            for stage_name, metrics in stages.items():
                plot_data.append({
                    'Missing_Rate': missing_rate,
                    'Method': method,
                    'Stage': stage_name,
                    'MAE': metrics['MAE'],
                    'R2': metrics['R2'],
                    'MAE_deg': metrics.get('MAE_deg_%', 0)
                })
    
    df_plot = pd.DataFrame(plot_data)
    
    # 1. Stage 3 MAE by method
    ax1 = axes[0, 0]
    stage3_data = df_plot[df_plot['Stage'] == 'Stage3']
    sns.barplot(data=stage3_data, x='Missing_Rate', y='MAE', hue='Method', ax=ax1)
    ax1.set_title('Final Prediction (Stage 3) - MAE', fontweight='bold')
    ax1.set_ylabel('MAE')
    
    # 2. Stage 3 R² by method
    ax2 = axes[0, 1]
    sns.barplot(data=stage3_data, x='Missing_Rate', y='R2', hue='Method', ax=ax2)
    ax2.set_title('Final Prediction (Stage 3) - R²', fontweight='bold')
    ax2.set_ylabel('R² Score')
    
    # 3. All stages comparison for DLPIM at 40%
    ax3 = axes[1, 0]
    dlpim_40 = df_plot[(df_plot['Method'] == 'DLPIM') & (df_plot['Missing_Rate'] == '40%')]
    if len(dlpim_40) > 0:
        stages = ['Stage1', 'Stage2', 'Stage3']
        mae_values = [dlpim_40[dlpim_40['Stage'] == s]['MAE'].values[0] for s in stages]
        ax3.bar(stages, mae_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax3.set_title('DLPIM Progressive Refinement (40% missing)', fontweight='bold')
        ax3.set_ylabel('MAE')
    
    # 4. Degradation comparison
    ax4 = axes[1, 1]
    stage3_deg = stage3_data[stage3_data['Missing_Rate'] == '40%']
    if len(stage3_deg) > 0:
        sns.barplot(data=stage3_deg, x='Method', y='MAE_deg', ax=ax4)
        ax4.set_title('Performance Degradation at 40% Missing', fontweight='bold')
        ax4.set_ylabel('MAE Increase (%)')
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('Figure_4.8.1_MultiStage_Performance.pdf', dpi=300, bbox_inches='tight')
    print("✓ Figure 4.8.1 saved: Figure_4.8.1_MultiStage_Performance.pdf")
    plt.show()


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    results = run_multistage_prediction('Nutrients_No_missing_arranged_in_pattern.csv')
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. Table_4.8.1_MultiStage_Results.xlsx")
    print("  2. Figure_4.8.1_MultiStage_Performance.pdf")