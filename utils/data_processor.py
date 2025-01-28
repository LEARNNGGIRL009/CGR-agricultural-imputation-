import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

def create_missing_pattern(data, rate):
    """
    Create missing pattern ensuring CGR column has no missing values.
    CGR (last column) is used as guiding information for imputation.
    
    Parameters:
    data : numpy array
        Input data array
    rate : float
        Missing rate (between 0 and 1)
    
    Returns:
    mask : numpy array (boolean)
        Mask indicating missing values (False) and observed values (True)
    """
    # can you based on weekly data collection as well
    '''
    def create_missing_pattern(data, missing_weeks):
    mask = np.ones(len(data), dtype=bool)
    week_length = 76
    for week in range(missing_weeks):
        start_idx = week * week_length
        end_idx = start_idx + week_length
        mask[start_idx:end_idx] = False
    return mask

    '''
    # Initialize mask as boolean array with all True values
    mask = np.ones_like(data, dtype=bool)
    
    # Create missing pattern only for feature columns (excluding CGR)
    for i in range(data.shape[1]-1):  # Exclude CGR column
        missing_idx = np.random.choice(
            data.shape[0],
            size=int(data.shape[0] * rate),
            replace=False
        )
        mask[missing_idx, i] = False  # Mark as missing
    
    # CGR column (last column) remains completely observed (all True)
    mask[:, -1] = True  # Ensure CGR column has no missing values
    
    # Verify missing rate for non-CGR columns
    actual_rate = (~mask[:, :-1]).sum() / (data.shape[0] * (data.shape[1]-1))
    print(f"Actual missing rate for features: {actual_rate:.2%}")
    print(f"CGR column missing rate: 0.00%")
    
    return mask