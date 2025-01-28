#class DeepImputationBase:

import numpy as np
import torch
import torch.nn as nn

class DeepImputationBase:
    """Base class for deep learning imputation methods"""
    def __init__(self):
        self.is_fitted = False
    
    def _validate_input(self, X, mask=None):
        """Validate input data and mask"""
        X = np.asarray(X)
        if mask is None:
            mask = ~np.isnan(X)
        else:
            mask = np.asarray(mask)
            
        if X.shape != mask.shape:
            raise ValueError(f"X shape {X.shape} does not match mask shape {mask.shape}")
            
        return X, mask
    
    def _check_missing_pattern(self, mask):
        """Analyze missing data pattern"""
        missing_rate = (~mask).mean()
        missing_cols = (~mask).any(axis=0)
        return {
            'missing_rate': missing_rate,
            'columns_with_missing': missing_cols.sum(),
            'pattern': 'MCAR'  # Can be extended for MAR/MNAR analysis
        }


class BaseImputer:
    """Base class for all imputation methods"""
    def __init__(self):
        self.is_fitted = False
    
    def _validate_input(self, X, mask=None):
        """Validate input data and mask"""
        X = np.asarray(X)
        if mask is None:
            mask = ~np.isnan(X)
        else:
            mask = np.asarray(mask)
            
        if X.shape != mask.shape:
            raise ValueError(f"X shape {X.shape} does not match mask shape {mask.shape}")
            
        return X, mask
    
    def _check_missing_pattern(self, mask):
        """Analyze missing data pattern"""
        missing_rate = (~mask).mean()
        missing_cols = (~mask).any(axis=0)
        return {
            'missing_rate': missing_rate,
            'columns_with_missing': missing_cols.sum(),
            'pattern': 'MCAR'  # Can be extended for MAR/MNAR analysis
        }
        