import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import warnings
warnings.filterwarnings('ignore')

# Path settings
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "raw" / "train.csv"
SCALER_FILE = BASE_DIR / "data" / "interim" / "std_scaler_v1.joblib"
MODEL_FILE = BASE_DIR / "data" / "interim" / "rf_model_v1.joblib"

# Data preparation
df = pd.read_csv(DATA_FILE, encoding="utf-8", low_memory=False)

feature_columns = [
    "cast_pressure",
    "count", 
    "upper_mold_temp1",
    "low_section_speed",
    "lower_mold_temp2",
    "high_section_speed",
    "upper_mold_temp2",
    "lower_mold_temp1",
    "biscuit_thickness",
    "sleeve_temperature"
]

X_original = df[feature_columns].copy()
X_original = X_original.dropna()

# Load models
std_scaler = joblib.load(SCALER_FILE)
rf_model = joblib.load(MODEL_FILE)

# Data preprocessing  
X_scaled = std_scaler.transform(X_original)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns, index=X_original.index)

# Matplotlib settings
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.style.use('default')

# Create 2x5 subplots
fig, axes = plt.subplots(figsize=(20, 12), nrows=2, ncols=5)
fig.suptitle('Partial Dependence Plot - Feature Impact on Prediction\n(Random Forest Model)', 
             fontsize=16, fontweight='bold', y=0.98)

axes_flat = axes.flatten()

# Sampling for efficiency - use scaled data as model expects
sample_size = min(1000, len(X_scaled_df))
if len(X_scaled_df) > sample_size:
    X_sample = X_scaled_df.sample(n=sample_size, random_state=42)
else:
    X_sample = X_scaled_df

# Generate PDP for each variable
for i, feature in enumerate(feature_columns):
    try:
        display = PartialDependenceDisplay.from_estimator(
            estimator=rf_model,
            X=X_sample,
            features=[i],
            feature_names=feature_columns,
            percentiles=(0.05, 0.95),
            n_jobs=1,
            random_state=42,
            ax=axes_flat[i]
        )
        
        axes_flat[i].set_title(f'{feature}', fontsize=12, fontweight='bold', pad=25)
        axes_flat[i].set_xlabel('Scaled Feature Value', fontsize=10)
        axes_flat[i].set_ylabel('Partial Dependence', fontsize=10)
        axes_flat[i].grid(True, alpha=0.3)
        axes_flat[i].margins(y=0.1)
        
    except Exception:
        axes_flat[i].text(0.5, 0.5, f'{feature}\n(Failed)', 
                         ha='center', va='center', transform=axes_flat[i].transAxes,
                         fontsize=10, color='red')
        axes_flat[i].set_title(f'{feature}', fontsize=12, fontweight='bold', pad=25)

# Layout optimization
plt.tight_layout()
plt.subplots_adjust(top=0.85, hspace=0.5, wspace=0.3)

# Show plot only
plt.show()

# Clean up memory
plt.close(fig)

# hello