import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# generatinf uniform and non uniform missing rates to implement multivariate time-series

def create_nonuniform_missing_pattern(data, rate):
    np.random.seed(42)
    mask = np.ones(len(data), dtype=bool)
    total_rows = len(data)
    
    missing_points = int(total_rows * rate)
    # Create more missing values in early weeks
    weights = np.linspace(1.5, 0.5, total_rows)
    weights = weights / np.sum(weights)
    missing_idx = np.random.choice(total_rows, size=missing_points, replace=False, p=weights)
    mask[missing_idx] = False
    return mask

data = pd.read_csv('D://From One drive//Paper 001//Paper 1//DLPIMjan2stoutput//Nutrients_No_missing_arranged_in_pattern.csv')

fig, axes = plt.subplots(6, 1, figsize=(15, 20))
missing_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

for idx, rate in enumerate(missing_rates):
    mask = create_nonuniform_missing_pattern(data['Na'], rate)
    data_missing = data['Na'].copy()
    data_missing[~mask] = np.nan
    
    axes[idx].plot(range(len(data_missing)), data_missing, '.', markersize=1, color='blue', alpha=0.5)
    axes[idx].set_ylim(0, 2)
    axes[idx].set_ylabel(f'Missing rate: {int(rate*100)}%')
    
    week_ticks = np.arange(0, len(data), 76)
    axes[idx].set_xticks(week_ticks + 76/2)
    axes[idx].set_xticklabels([f'week{i+1}' for i in range(13)], rotation=45)

axes[-1].set_xlabel('Week')
plt.tight_layout()
plt.savefig('na_nonuniform_pattern.tiff', 
            dpi=600, bbox_inches='tight')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_uniform_missing_pattern(data, rate):
    np.random.seed(42)
    mask = np.ones(len(data), dtype=bool)
    week_length = 76
    num_weeks = len(data) // week_length
    
    points_per_week = int(week_length * rate)
    for week in range(num_weeks):
        start_idx = week * week_length
        week_indices = np.arange(start_idx, start_idx + week_length)
        missing_indices = np.random.choice(week_indices, points_per_week, replace=False)
        mask[missing_indices] = False
    return mask

data = pd.read_csv('data.csv')

fig, axes = plt.subplots(6, 1, figsize=(15, 20))
missing_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

for idx, rate in enumerate(missing_rates):
    mask = create_uniform_missing_pattern(data['Na'], rate)
    data_missing = data['Na'].copy()
    data_missing[~mask] = np.nan
    
    axes[idx].plot(range(len(data_missing)), data_missing, '.', markersize=1, color='blue', alpha=0.5)
    axes[idx].set_ylim(0, 2)
    axes[idx].set_ylabel(f'Missing rate: {int(rate*100)}%')
    
    week_ticks = np.arange(0, len(data), 76)
    axes[idx].set_xticks(week_ticks + 76/2)
    axes[idx].set_xticklabels([f'week{i+1}' for i in range(13)], rotation=45)

axes[-1].set_xlabel('Week')
plt.tight_layout()
plt.savefig('na__uniform_pattern.tiff', 
            dpi=600, bbox_inches='tight')
plt.show()