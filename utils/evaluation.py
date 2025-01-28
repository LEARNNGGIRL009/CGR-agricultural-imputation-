import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def comprehensive_evaluation(original, imputed, mask):
    """Enhanced evaluation metrics"""
    missing_idx = ~mask
    metrics = {
        'MSE': mean_squared_error(original[missing_idx], imputed[missing_idx]),
        'MAE': mean_absolute_error(original[missing_idx], imputed[missing_idx]),
        'R2': r2_score(original[missing_idx], imputed[missing_idx]),
        'RMSE': np.sqrt(mean_squared_error(original[missing_idx], imputed[missing_idx])),
        'max_error': np.max(np.abs(original[missing_idx] - imputed[missing_idx])),
        'normalized_mae': mean_absolute_error(original[missing_idx], imputed[missing_idx]) / np.std(original[missing_idx])
    }
    
    # Add column-wise metrics
    col_metrics = {}
    for col in range(original.shape[1]):
        col_mask = missing_idx[:, col]
        if np.any(col_mask):
            col_metrics[f'col_{col}_mae'] = mean_absolute_error(
                original[col_mask, col],
                imputed[col_mask, col]
            )
    
    metrics.update(col_metrics)
    return metrics

def analyze_cgr_correlations(original_data, imputed_data, mask):
    """Analyze correlations between imputed values and CGR patterns"""
    cgr = original_data[:, -1]  # CGR is the last column
    results = {}
    
    # Feature-wise correlation analysis
    for i in range(original_data.shape[1]-1):  # Exclude CGR column
        missing_idx = ~mask[:, i]
        if np.any(missing_idx):
            # Correlation for imputed values
            imp_corr = np.corrcoef(imputed_data[missing_idx, i], cgr[missing_idx])[0,1]
            # Correlation for original values
            orig_corr = np.corrcoef(original_data[missing_idx, i], cgr[missing_idx])[0,1]
            
            results[f'feature_{i}'] = {
                'imputed_cgr_correlation': imp_corr,
                'original_cgr_correlation': orig_corr,
                'correlation_preservation': abs(imp_corr - orig_corr)
            }
    
    return results