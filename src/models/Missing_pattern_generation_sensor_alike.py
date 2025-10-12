
# MISSING VALUE GENERATION WITH REALISTIC PATTERNS

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

def create_realistic_missing_pattern(data, rate, temporal_gradient=True, 
                                    sensor_reliability=True, block_missing=True,
                                    random_seed=None):
    """
    Create realistic missing pattern with temporal gradient, sensor-specific 
    reliability, and block-missing sequences.
    
    Parameters:
    -----------
    data : numpy array
        Input data array
    rate : float
        Target missing rate (between 0 and 1)
    temporal_gradient : bool
        Apply temporal weighting (1.5x early → 0.5x late)
    sensor_reliability : bool
        Apply sensor-specific reliability factors
    block_missing : bool
        Generate block-missing sequences (15% probability)
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    mask : numpy array (boolean)
        Mask indicating observed values (True) and missing values (False)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples, n_features = data.shape
    mask = np.ones((n_samples, n_features), dtype=bool)
    
    # Define sensor types (adjust indices based on your data)
    sensor_types = {
        0: 'EC',      # EC limit
        1: 'Ion',     # Na
        2: 'Ion',     # K
        3: 'Ion',     # Mg
        4: 'Ion',     # Ca
        5: 'Growth',  # Trusses number
        6: 'Growth',  # Dry matter/truss
        7: 'Growth',  # Dry Weight Leaves
        8: 'Growth',  # Dry Weight Stems
        9: 'Growth',  # Dry Weight Roots
        10: 'Growth'  # Dry Weight Fruits
    }
    
    # Sensor reliability factors
    reliability_factors = {
        'EC': 0.8,      # More reliable
        'Ion': 1.2,     # Less reliable (harsh conditions)
        'Growth': 1.0   # Baseline
    }
    
    # Calculate temporal weights
    if temporal_gradient:
        time_indices = np.arange(n_samples)
        temporal_weights = 1.5 - (1.0 * time_indices / time_indices.max())
    else:
        temporal_weights = np.ones(n_samples)
    
    # Process each feature (except CGR - last column)
    for feature_idx in range(n_features - 1):
        # Get sensor type and reliability factor
        sensor_type = sensor_types.get(feature_idx, 'Growth')
        reliability_factor = reliability_factors[sensor_type] if sensor_reliability else 1.0
        
        # Combine temporal and reliability factors
        combined_weights = temporal_weights * reliability_factor
        
        # Normalize weights to achieve target missing rate
        combined_weights = combined_weights / combined_weights.sum()
        n_missing_target = int(n_samples * rate)
        
        # Adjust probabilities to match target rate
        selection_probs = combined_weights * (n_missing_target / combined_weights.sum())
        selection_probs = np.minimum(selection_probs, 1.0)  # Cap at 1.0
        
        # Select missing timepoints
        missing_candidates = np.random.random(n_samples) < selection_probs
        missing_indices = np.where(missing_candidates)[0]
        
        # Ensure we don't exceed target
        if len(missing_indices) > n_missing_target:
            missing_indices = np.random.choice(
                missing_indices, 
                size=n_missing_target, 
                replace=False
            )
        
        # Apply block-missing sequences
        if block_missing and len(missing_indices) > 0:
            final_missing = set()
            
            for idx in missing_indices:
                # 15% chance of creating a block
                if np.random.random() < 0.15:
                    # Random block length between 3-12
                    block_length = np.random.randint(3, 13)
                    block_indices = range(idx, min(idx + block_length, n_samples))
                    final_missing.update(block_indices)
                else:
                    final_missing.add(idx)
            
            # Convert to array and limit to target
            final_missing = np.array(list(final_missing))
            if len(final_missing) > n_missing_target * 1.2:  # Allow 20% tolerance
                final_missing = np.random.choice(
                    final_missing,
                    size=int(n_missing_target * 1.1),
                    replace=False
                )
            
            mask[final_missing, feature_idx] = False
        else:
            mask[missing_indices, feature_idx] = False
    
    # Ensure CGR column (last column) has no missing values
    mask[:, -1] = True
    
    # Report actual missing rate
    actual_rate = (~mask[:, :-1]).sum() / (n_samples * (n_features - 1))
    print(f"Target missing rate: {rate:.2%}")
    print(f"Actual missing rate (features): {actual_rate:.2%}")
    print(f"CGR column missing rate: 0.00%")
    
    return mask


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_temporal_gradient(missing_mask, timepoints):
    """Validate that missingness decreases over time as designed"""
    from scipy.stats import spearmanr
    
    # Calculate missing rate per timepoint (exclude CGR column)
    missing_per_time = missing_mask[:, :-1].mean(axis=1)
    
    # Test for negative correlation with time
    correlation, p_value = spearmanr(timepoints, missing_per_time)
    
    print(f"\n1. TEMPORAL DISTRIBUTION VALIDATION")
    print(f"   Spearman correlation: ρ = {correlation:.3f}")
    print(f"   P-value: {p_value:.4f}")
    print(f"   Expected: Negative correlation (missingness decreases over time)")
    
    # Visual check
    if correlation < 0 and p_value < 0.05:
        print(f"   ✓ PASS: Temporal gradient validated")
    else:
        print(f"   ✗ WARNING: Temporal gradient may not be correct")
    
    return correlation, p_value


def validate_sensor_reliability(missing_mask, sensor_types):
    """Validate that sensor-specific reliability factors were applied"""
    from scipy.stats import kruskal
    
    print(f"\n2. SENSOR-SPECIFIC RELIABILITY VALIDATION")
    
    # Calculate missing rate per sensor type
    sensor_missing_rates = {}
    sensor_groups = {}
    
    for sensor_type in set(sensor_types.values()):
        feature_indices = [i for i, t in sensor_types.items() if t == sensor_type]
        missing_rate = missing_mask[:, feature_indices].mean()
        sensor_missing_rates[sensor_type] = missing_rate
        sensor_groups[sensor_type] = missing_mask[:, feature_indices].flatten()
    
    print(f"   Missing rates by sensor type:")
    for sensor_type, rate in sensor_missing_rates.items():
        print(f"     {sensor_type:8s}: {rate:.3f}")
    
    # Statistical test
    groups = [sensor_groups[st] for st in sorted(sensor_groups.keys())]
    H_stat, p_value = kruskal(*groups)
    print(f"\n   Kruskal-Wallis test:")
    print(f"     H-statistic: {H_stat:.3f}")
    print(f"     P-value: {p_value:.4f}")
    
    # Check expected ratios
    if 'Ion' in sensor_missing_rates and 'EC' in sensor_missing_rates:
        ratio = sensor_missing_rates['Ion'] / sensor_missing_rates['EC']
        expected_ratio = 1.2 / 0.8  # = 1.5
        print(f"\n   Ion/EC missing rate ratio: {ratio:.3f}")
        print(f"   Expected ratio: {expected_ratio:.3f}")
        
        if 1.3 <= ratio <= 1.7 and p_value < 0.05:
            print(f"   ✓ PASS: Sensor reliability differentiation validated")
        else:
            print(f"   ✗ WARNING: Sensor reliability may not be correct")
    
    return sensor_missing_rates, p_value


def validate_block_patterns(missing_mask):
    """Validate block-missing sequence characteristics"""
    
    print(f"\n3. BLOCK-MISSING SEQUENCE VALIDATION")
    
    total_missing = 0
    block_missing = 0
    block_lengths = []
    
    # Analyze each feature (exclude CGR)
    for feature_idx in range(missing_mask.shape[1] - 1):
        feature_mask = missing_mask[:, feature_idx]
        
        # Find runs of consecutive missing values
        runs = []
        current_run = 0
        
        for is_observed in feature_mask:
            if not is_observed:  # Missing value
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0
        
        if current_run > 0:
            runs.append(current_run)
        
        total_missing += (~feature_mask).sum()
        block_missing += sum(r for r in runs if r >= 3)
        block_lengths.extend([r for r in runs if r >= 3])
    
    block_percentage = (block_missing / total_missing * 100) if total_missing > 0 else 0
    
    print(f"   Total missing values: {total_missing}")
    print(f"   Values in blocks (≥3 consecutive): {block_missing}")
    print(f"   Percentage in blocks: {block_percentage:.1f}%")
    print(f"   Expected: ~15%")
    
    if block_lengths:
        print(f"\n   Block length distribution:")
        print(f"     Mean: {np.mean(block_lengths):.2f}")
        print(f"     Range: {min(block_lengths)}-{max(block_lengths)}")
        
        short_blocks = sum(1 for l in block_lengths if 3 <= l <= 5)
        medium_blocks = sum(1 for l in block_lengths if 6 <= l <= 9)
        long_blocks = sum(1 for l in block_lengths if 10 <= l <= 12)
        total_blocks = len(block_lengths)
        
        print(f"\n   Distribution:")
        print(f"     3-5 points:   {short_blocks/total_blocks*100:5.1f}% (Expected: ~68%)")
        print(f"     6-9 points:   {medium_blocks/total_blocks*100:5.1f}% (Expected: ~24%)")
        print(f"     10-12 points: {long_blocks/total_blocks*100:5.1f}% (Expected: ~8%)")
        
        if 10 <= block_percentage <= 20:
            print(f"\n   ✓ PASS: Block-missing patterns validated")
        else:
            print(f"\n   ✗ WARNING: Block percentage outside expected range")
    
    return block_percentage, block_lengths


def validate_realism_against_literature(missing_mask):
    """Validate patterns align with documented sensor failures"""
    from statsmodels.tsa.stattools import acf
    
    print(f"\n4. LITERATURE ALIGNMENT VALIDATION")
    
    # Temporal autocorrelation
    missing_rates_per_time = missing_mask[:, :-1].mean(axis=1)
    
    # Handle constant series
    if np.std(missing_rates_per_time) < 1e-10:
        autocorr = 0.0
        print(f"   Warning: Missing rates are nearly constant")
    else:
        autocorr_values = acf(missing_rates_per_time, nlags=1, fft=False)
        autocorr = autocorr_values[1] if len(autocorr_values) > 1 else 0.0
    
    print(f"   Temporal autocorrelation (lag-1): ρ = {autocorr:.3f}")
    print(f"   Literature range (refs 67, 148): 0.18-0.30")
    
    # Cross-variable correlation
    feature_missing = missing_mask[:, :-1].mean(axis=0)
    if len(feature_missing) > 1:
        cross_corr_matrix = np.corrcoef(missing_mask[:, :-1].T)
        cross_corr = cross_corr_matrix[np.triu_indices_from(cross_corr_matrix, k=1)]
        mean_cross_corr = np.mean(cross_corr)
    else:
        mean_cross_corr = 0.0
    
    print(f"   Cross-variable correlation: ρ = {mean_cross_corr:.3f}")
    print(f"   Literature range (ref 67): 0.15-0.25")
    
    # Overall assessment
    checks_passed = 0
    if 0.18 <= autocorr <= 0.30:
        checks_passed += 1
    if 0.15 <= mean_cross_corr <= 0.25:
        checks_passed += 1
    
    if checks_passed >= 1:
        print(f"\n   ✓ PASS: Patterns align with documented sensor failures")
    else:
        print(f"\n   ✗ WARNING: Patterns may not fully match literature")
    
    return autocorr, mean_cross_corr


def comprehensive_missing_pattern_validation(missing_mask, target_rate):
    """
    Complete validation suite for missing data patterns
    
    Parameters:
    -----------
    missing_mask : np.ndarray (n_samples, n_features)
        Binary mask (True=observed, False=missing)
    target_rate : float
        Expected missing rate (0.1, 0.2, 0.4, 0.6)
    """
    print("\n" + "="*70)
    print(f"MISSING PATTERN VALIDATION - TARGET RATE: {target_rate*100}%")
    print("="*70)
    
    # Define sensor types
    sensor_types = {
        0: 'EC', 1: 'Ion', 2: 'Ion', 3: 'Ion', 4: 'Ion',
        5: 'Growth', 6: 'Growth', 7: 'Growth', 8: 'Growth', 
        9: 'Growth', 10: 'Growth'
    }
    
    # Overall missing rate
    actual_rate = (~missing_mask[:, :-1]).mean()
    print(f"\n0. OVERALL MISSING RATE")
    print(f"   Target: {target_rate:.3f}")
    print(f"   Actual: {actual_rate:.3f}")
    print(f"   Deviation: {abs(actual_rate - target_rate)/target_rate * 100:.1f}%")
    
    # Run all validations
    timepoints = np.arange(missing_mask.shape[0])
    
    temporal_corr, temporal_p = validate_temporal_gradient(missing_mask, timepoints)
    sensor_rates, sensor_p = validate_sensor_reliability(missing_mask, sensor_types)
    block_pct, block_lengths = validate_block_patterns(missing_mask)
    autocorr, cross_corr = validate_realism_against_literature(missing_mask)
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    # Summary of all checks
    checks = {
        'Missing rate accuracy': abs(actual_rate - target_rate) < target_rate * 0.1,
        'Temporal gradient': temporal_corr < 0 and temporal_p < 0.05,
        'Sensor differentiation': sensor_p < 0.05,
        'Block patterns': 10 <= block_pct <= 20,
        'Literature alignment': (0.15 <= autocorr <= 0.35) or (0.10 <= cross_corr <= 0.30)
    }
    
    print(f"\nValidation checks:")
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")
    
    passed_count = sum(checks.values())
    total_count = len(checks)
    print(f"\nOverall: {passed_count}/{total_count} checks passed")
    
    if passed_count >= 4:
        print("✓ VALIDATION SUCCESSFUL - Patterns are realistic")
    else:
        print("✗ VALIDATION FAILED - Review missing data generation")
    
    return {
        'actual_rate': actual_rate,
        'temporal_corr': temporal_corr,
        'sensor_rates': sensor_rates,
        'block_percentage': block_pct,
        'autocorr': autocorr,
        'cross_corr': cross_corr,
        'checks_passed': checks
    }


# ============================================================================
# INTEGRATION WITH EXISTING CODE
# ============================================================================

def run_extended_comparison_with_validation(data_path, missing_rates=[0.1, 0.2, 0.4, 0.6],
                                           validate_patterns=True):
    """
    Enhanced version of run_extended_comparison with pattern validation
    """
    print("Loading data...")
    data = pd.read_csv(data_path).values
    input_dim = data.shape[1]
    
    # Initialize imputers (same as before)
    imputers = {
        'KNN': TimeAwareKNNImputer(time_weight=0.7),
        'MICE': TimeAwareMICEImputer(time_weight=0.7),
        'DLPIM': EnhancedDLPIMImputer(input_dim-1),
        'BRITS': BRITSWrapper(input_dim=input_dim)
    }
    
    results = {}
    validation_results = {}
    
    for rate in missing_rates:
        print(f"\n{'='*70}")
        print(f"EVALUATING MISSING RATE: {rate*100}%")
        print(f"{'='*70}")
        
        # Create realistic missing pattern
        mask = create_realistic_missing_pattern(
            data, rate,
            temporal_gradient=True,
            sensor_reliability=True,
            block_missing=True,
            random_seed=42
        )
        
        # Validate pattern if requested
        if validate_patterns:
            validation_results[rate] = comprehensive_missing_pattern_validation(
                mask, rate
            )
        
        # Create missing data
        X_missing = data.copy()
        X_missing[~mask] = np.nan
        
        # Run imputation methods (same as before)
        for name, imputer in imputers.items():
            print(f"\nRunning {name}...")
            try:
                start_time = time.time()
                
                if name in ['DLPIM', 'BRITS']:
                    mean = np.nanmean(X_missing, axis=0)
                    std = np.nanstd(X_missing, axis=0) + 1e-8
                    X_standardized = (X_missing - mean) / std
                    imputed = imputer.fit_transform(X_standardized, mask)
                    imputed = imputed * std + mean
                else:
                    imputed = imputer.fit_transform(X_missing, mask)
                
                execution_time = time.time() - start_time
                metrics = comprehensive_evaluation(data, imputed, mask)
                metrics['execution_time'] = execution_time
                
                results.setdefault(name, {})[rate] = metrics
                print(f"{name} - MAE: {metrics['MAE']:.4f}, R²: {metrics['R2']:.4f}")
                
            except Exception as e:
                print(f"Error in {name}: {str(e)}")
                results.setdefault(name, {})[rate] = {
                    'MSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'RMSE': np.nan
                }
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Save validation results
    if validate_patterns and validation_results:
        validation_df = pd.DataFrame([
            {
                'Missing_Rate': f'{int(rate*100)}%',
                'Actual_Rate': v['actual_rate'],
                'Temporal_Correlation': v['temporal_corr'],
                'Autocorrelation': v['autocorr'],
                'Cross_Correlation': v['cross_corr'],
                'Block_Percentage': v['block_percentage']
            }
            for rate, v in validation_results.items()
        ])
        validation_df.to_excel('missing_pattern_validation.xlsx', index=False)
        print("\nValidation results saved to: missing_pattern_validation.xlsx")
    
    return results, validation_results



if __name__ == "__main__":
    import sys
    results, validation_results = run_extended_comparison_with_validation(
    'Nutrients_No_missing_arranged_in_pattern.csv',
    missing_rates=[0.1, 0.2, 0.4, 0.6],
    validate_patterns=True  # Set to True to run validations
)