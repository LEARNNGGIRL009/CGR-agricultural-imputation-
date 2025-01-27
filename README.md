# CGR-Guided Agricultural Data Imputation Methods

This repository contains the implementation of novel imputation methods for agricultural data with Crop Growth Rate (CGR) guidance. The methods include Enhanced GAIN, DLPIM, BRITS, and other traditional imputation techniques.

This research presents novel approaches to missing data imputation in agricultural datasets, specifically focusing on leveraging Crop Growth Rate (CGR) as a guiding factor. We introduce several enhanced imputation methods that demonstrate superior performance in maintaining agricultural data patterns and relationships.

## Methods Implemented
- TimeAwareKNNImputer
- TimeAwareMICEImputer
- TimeAwareMissForest
- SoftImputeWrapper
- AutoencoderImputer
- EMImputer
- EnhancedGAIN
- DEGAINImputer
- EnhancedDLPIMImputer
- BRITSWrapper

## Installation

```
pip install -r requirements.txt
```

## Data Format
The input data should be in CSV format with the following structure:
- The last column should be the CGR (Crop Growth Rate)
- All other columns represent agricultural features
- Missing values should be represented as NaN

## Usage

Basic usage example:
```python
from src.models import EnhancedDLPIMImputer
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('your_data.csv').values
input_dim = data.shape[1]

# Create missing pattern
mask = create_missing_pattern(data, rate=0.2)  # 20% missing rate

# Initialize imputer
imputer = EnhancedDLPIMImputer(input_dim=input_dim-1)  # -1 because CGR is handled separately

# Perform imputation
imputed_data = imputer.fit_transform(data, mask)
```

## Reproducing Results

To reproduce the experiments from our paper:

```
python src/experiments/run_comparison.py
```

This will:
1. Load the dataset
2. Create various missing patterns
3. Apply all imputation methods
4. Generate comparison metrics and plots

## Repository Structure
```
project-root/
├── src/
│   ├── models/              # Implementation of all imputation methods
│   ├── utils/               # Utility functions and metrics
│   └── experiments/         # Experiment scripts
├── data/                    # Data directory
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Dependencies
- Python 3.8+
- PyTorch 1.9+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{author2024cgr,
  title={CGR-Guided Agricultural Data Imputation},
  author={Viji venugopal},
  journal={Journal Name}, #update soon
  year={2025}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
