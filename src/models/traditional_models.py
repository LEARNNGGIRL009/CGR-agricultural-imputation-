
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from ..utils.data_processor import create_missing_pattern

class TimeAwareKNNImputer:
    def __init__(self, n_neighbors=15, time_weight=0.7):
        self.n_neighbors = n_neighbors
        self.time_weight = time_weight
    
    def _compute_temporal_distances(self, X):
        """Compute temporal distances between samples"""
        n_samples = X.shape[0]
        temporal_dist = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                # Higher weight for temporally distant samples
                temporal_dist[i, j] = abs(i - j) * self.time_weight
        return temporal_dist
    
    def fit_transform(self, X, mask=None):
        if mask is None:
            mask = ~np.isnan(X)
        
        X_imp = X.copy()
        temporal_dist = self._compute_temporal_distances(X)
        
        # Initial mean imputation
        means = np.nanmean(X_imp, axis=0)
        for i in range(X.shape[1]):
            X_imp[~mask[:, i], i] = means[i]
        
        # Impute considering temporal distances
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if not mask[i, j]:
                    # Find temporal neighbors
                    distances = temporal_dist[i]
                    # Add feature distances
                    feature_dist = np.linalg.norm(X_imp - X_imp[i], axis=1)
                    total_dist = distances + (1 - self.time_weight) * feature_dist
                    
                    # Get k nearest neighbors
                    neighbor_indices = np.argsort(total_dist)[:self.n_neighbors]
                    X_imp[i, j] = np.mean(X_imp[neighbor_indices, j])
        
        return X_imp

class TimeAwareMICEImputer:
    def __init__(self, max_iter=15, time_weight=0.7):
        self.max_iter = max_iter
        self.time_weight = time_weight
        
    def fit_transform(self, X, mask=None):
        if mask is None:
            mask = ~np.isnan(X)
            
        X_imp = X.copy()
        n_samples = X.shape[0]
        
        # Add temporal features
        time_indices = np.arange(n_samples).reshape(-1, 1)
        X_with_time = np.hstack([X_imp, time_indices * self.time_weight])
        
        # Initial mean imputation
        means = np.nanmean(X_imp, axis=0)
        for i in range(X.shape[1]):
            X_imp[~mask[:, i], i] = means[i]
            
        for _ in range(self.max_iter):
            X_old = X_imp.copy()
            
            for col in range(X.shape[1]):
                missing_idx = ~mask[:, col]
                if np.any(missing_idx):
                    # Consider temporal order in regression
                    reg = LinearRegression()
                    observed_idx = mask[:, col]
                    
                    # Include time index in features
                    features = np.column_stack([
                        X_imp[:, [i for i in range(X.shape[1]) if i != col]],
                        time_indices * self.time_weight
                    ])
                    
                    reg.fit(features[observed_idx], X_imp[observed_idx, col])
                    X_imp[missing_idx, col] = reg.predict(features[missing_idx])
            
            if np.allclose(X_old, X_imp, rtol=1e-5):
                break
                
        return X_imp
        
class SoftImputeWrapper:
    def __init__(self, rank=10, lambda_=0.1, max_iter=100, tol=1e-5):
        self.rank = rank
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        
    def _soft_thresh(self, x, lambda_):
        """Soft thresholding operator"""
        return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)
    
    def _initialize_missing(self, X, mask):
        """Better initialization for missing values"""
        # Initialize with column means
        col_mean = np.nanmean(X, axis=0)
        X_filled = X.copy()
        for i in range(X.shape[1]):
            missing_idx = ~mask[:, i]
            X_filled[missing_idx, i] = col_mean[i]
        
        # Add noise to avoid exact zero eigenvalues
        noise = np.random.normal(0, 0.01, X_filled.shape)
        X_filled[~mask] += noise[~mask]
        return X_filled
    
    def fit_transform(self, X, mask=None):
        if mask is None:
            mask = ~np.isnan(X)
        
        # Scale the data
        self.col_means = np.nanmean(X, axis=0)
        self.col_stds = np.nanstd(X, axis=0) + 1e-8
        X_scaled = (X - self.col_means) / self.col_stds
        
        # Initialize missing values
        X_filled = self._initialize_missing(X_scaled, mask)
        X_old = X_filled.copy()
        
        # Iterative SVD with soft thresholding
        for iter_num in range(self.max_iter):
            # SVD
            try:
                U, s, Vt = np.linalg.svd(X_filled, full_matrices=False)
            except np.linalg.LinAlgError:
                print("SVD did not converge. Using current solution.")
                break
            
            # Soft thresholding of singular values
            s_thresh = self._soft_thresh(s, self.lambda_)
            
            # Reconstruct
            rank = min(self.rank, len(s_thresh))
            X_reconstruct = U[:, :rank] @ np.diag(s_thresh[:rank]) @ Vt[:rank, :]
            
            # Update only missing values
            X_filled[~mask] = X_reconstruct[~mask]
            
            # Check convergence
            change = np.linalg.norm(X_filled - X_old) / np.linalg.norm(X_old)
            if change < self.tol:
                break
                
            X_old = X_filled.copy()
        
        # Unscale the data
        X_filled = X_filled * self.col_stds + self.col_means
        return X_filled
        
class TimeAwareMissForest:
    def __init__(self, max_iter=10, time_weight=0.7):
        self.max_iter = max_iter
        self.time_weight = time_weight
        self.rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1,
            random_state=42
        )
    
    def fit_transform(self, X, mask=None):
        if mask is None:
            mask = ~np.isnan(X)
            
        X_imp = X.copy()
        n_samples = X.shape[0]
        
        # Add temporal features
        time_indices = np.arange(n_samples).reshape(-1, 1)
        
        # Initial mean imputation
        means = np.nanmean(X_imp, axis=0)
        for i in range(X.shape[1]):
            X_imp[~mask[:, i], i] = means[i]
        
        for _ in range(self.max_iter):
            X_old = X_imp.copy()
            
            for col in range(X.shape[1]):
                missing_idx = ~mask[:, col]
                if np.any(missing_idx):
                    # Include time index in features
                    features = np.column_stack([
                        X_imp[:, [i for i in range(X.shape[1]) if i != col]],
                        time_indices * self.time_weight
                    ])
                    
                    observed_idx = mask[:, col]
                    self.rf.fit(features[observed_idx], X_imp[observed_idx, col])
                    X_imp[missing_idx, col] = self.rf.predict(features[missing_idx])
            
            if np.allclose(X_old, X_imp, rtol=1e-5):
                break
                
        return X_imp

class EMImputer:
    def __init__(self, max_iter=100, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        
    def fit_transform(self, X, mask=None):
        if mask is None:
            mask = ~np.isnan(X)
            
        X_imp = X.copy()
        
        # Initial mean imputation
        means = np.nanmean(X_imp, axis=0)
        for i in range(X.shape[1]):
            X_imp[~mask[:, i], i] = means[i]
        
        for _ in range(self.max_iter):
            X_old = X_imp.copy()
            
            # E-step and M-step combined using linear regression
            for col in range(X.shape[1]):
                missing_idx = ~mask[:, col]
                if np.any(missing_idx):
                    other_cols = [i for i in range(X.shape[1]) if i != col]
                    
                    # Fit regression on observed data
                    reg = LinearRegression()
                    observed_idx = mask[:, col]
                    reg.fit(X_imp[observed_idx][:, other_cols], X_imp[observed_idx, col])
                    
                    # Impute missing values
                    X_imp[missing_idx, col] = reg.predict(X_imp[missing_idx][:, other_cols])
            
            # Check convergence
            if np.mean((X_old - X_imp) ** 2) < self.tol:
                break
                
        return X_imp  