import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn

class CGRAttentionGate:
    """Attention gate that uses CGR information to modulate feature attention"""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        # CGR processing network
        self.cgr_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, embed_dim),
            nn.Sigmoid()
        )
        
        # Feature attention network
        self.feature_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
    def forward(self, features, cgr):
        cgr = cgr.unsqueeze(-1) if cgr.dim() == 1 else cgr
        cgr_weights = self.cgr_net(cgr)
        feature_weights = self.feature_gate(features)
        combined_attention = cgr_weights * feature_weights
        return features * combined_attention

def analyze_cgr_correlations(original_data, imputed_data, mask):
    """Analyze correlations between imputed values and CGR patterns"""
    cgr = original_data[:, -1]
    results = {}
    
    for i in range(original_data.shape[1]-1):
        missing_idx = ~mask[:, i]
        if np.any(missing_idx):
            imp_corr = np.corrcoef(imputed_data[missing_idx, i], cgr[missing_idx])[0,1]
            orig_corr = np.corrcoef(original_data[missing_idx, i], cgr[missing_idx])[0,1]
            
            results[f'feature_{i}'] = {
                'imputed_cgr_correlation': imp_corr,
                'original_cgr_correlation': orig_corr,
                'correlation_preservation': abs(imp_corr - orig_corr)
            }
    
    return results

def process_cgr_features(data, window_size=24):
    """Process CGR features with temporal information"""
    cgr = data[:, -1]
    features = []
    
    for i in range(len(cgr) - window_size + 1):
        window = cgr[i:i + window_size]
        features.append([
            np.mean(window),
            np.std(window),
            np.min(window),
            np.max(window),
            np.median(window)
        ])
    
    return np.array(features)

def evaluate_cgr_consistency(original_data, imputed_data, mask):
    """Evaluate consistency of imputed values with CGR patterns"""
    cgr = original_data[:, -1]
    consistency_metrics = {}
    
    for i in range(original_data.shape[1]-1):
        missing_idx = ~mask[:, i]
        if np.any(missing_idx):
            # Correlation consistency
            orig_corr = np.corrcoef(original_data[missing_idx, i], cgr[missing_idx])[0,1]
            imp_corr = np.corrcoef(imputed_data[missing_idx, i], cgr[missing_idx])[0,1]
            
            # Value range consistency
            orig_range = np.ptp(original_data[missing_idx, i])
            imp_range = np.ptp(imputed_data[missing_idx, i])
            
            consistency_metrics[f'feature_{i}'] = {
                'correlation_preservation': abs(orig_corr - imp_corr),
                'range_preservation': abs(orig_range - imp_range) / orig_range
            }
    
    return consistency_metrics

def calculate_cgr_weighted_metrics(original_data, imputed_data, mask):
    """Calculate metrics weighted by CGR importance"""
    cgr = original_data[:, -1]
    cgr_weights = (cgr - cgr.min()) / (cgr.max() - cgr.min())  # Normalize CGR values
    
    weighted_metrics = {}
    for i in range(original_data.shape[1]-1):
        missing_idx = ~mask[:, i]
        if np.any(missing_idx):
            # Calculate weighted errors
            errors = np.abs(original_data[missing_idx, i] - imputed_data[missing_idx, i])
            weighted_mae = np.mean(errors * cgr_weights[missing_idx])
            weighted_mse = np.mean((errors ** 2) * cgr_weights[missing_idx])
            
            weighted_metrics[f'feature_{i}'] = {
                'weighted_mae': weighted_mae,
                'weighted_mse': weighted_mse,
                'weighted_rmse': np.sqrt(weighted_mse)
            }
    
    return weighted_metrics