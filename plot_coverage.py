import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2
from joblib import Parallel, delayed
from tqdm import tqdm
# Import your algorithms (make sure to use the vectorized LGR)
from src.algorithms import run_complete_randomization, run_acceptance_rejection, run_psrr, run_brain, run_lgr, run_lgr_adaptive, run_lgr_normalized, run_lgr_barrier
# Note: Assuming you added the function above to src.algorithms or defined it locally
# from src.algorithms import run_lgr_vectorized as run_lgr 
from src.data import generate_data
from src.estimator import DifferenceInMeans
from src.threshold import DefineThreshold

# ==========================================
# CONFIGURATION
# ==========================================
# Dimensions to test (The X-axis)
# We stop Rejection sampling early because it will hang forever at high d
d_values = [5, 10, 50, 100, 150, 200, 250] 
d_values = [10, 50, 100, 150, 200, 250] 
B = 1000  # Number of repetitions per dimension
B_frt = 100  # Number of FRT permutations
n = 500 # Total units
n1 = 250 # Treated units
p_val = 0.01 # Top 1% balance
n_jobs = -2  # Parallel cores
tau = 0.5 # True ATE
alpha = 0.05 # Significance level
distance_metric = "mahalanobis" # "mahalanobis" or "euclidean"
temperature = 0.5  # LGR temperature
eta = 1.0  # LGR eta parameter
linear = True # linear DGP

file = f'results/coverage_d{d_values}_metric{distance_metric}_B{B}_Bfrt{B_frt}_pval{p_val}_temperature{temperature}_eta{eta}_linear{linear}.pkl'

try:
    df = pd.read_pickle(file)
    print(f"Loaded existing results from {file}")
except FileNotFoundError:
    print("File not found, running new simulations...")
    print("Run the file 'competitor_simulation.py' to generate the results.")
    exit()


# Reshape for Seaborn (Wide to Long)
covered_cols = [c for c in df.columns if 'Covered' in c]
power_cols = [c for c in df.columns if 'Power' in c]

df_covered = pd.melt(df, id_vars=['d'], value_vars=covered_cols, 
                  var_name='Method', value_name='Covered')
df_power = pd.melt(df, id_vars=['d'], value_vars=power_cols, 
                  var_name='Method', value_name='Power')

# Clean names
df_covered['Method'] = df_covered['Method'].str.replace('_Covered', '')
df_power['Method'] = df_power['Method'].str.replace('_Power', '')

df_covered = df_covered.groupby(['d', 'Method'])['Covered'].mean().reset_index()
df_power = df_power.groupby(['d', 'Method'])['Power'].mean().reset_index()


# ==========================================
# PLOTTING COVERAGE & POWER
# ==========================================

sns.set_style("ticks")
plt.rcParams.update({
    'font.size': 20,          # General font size
    'axes.titlesize': 22,     # Subplot titles
    'axes.labelsize': 22,     # X and Y axis labels
    'xtick.labelsize': 16,    # X-axis tick labels
    'ytick.labelsize': 16,    # Y-axis tick labels
    'legend.fontsize': 13,    # Legend text
    'legend.title_fontsize': 13, # Legend title
    'lines.linewidth': 4,     # Thicker lines
    'lines.markersize': 15    # Larger markers
})

fig, axes = plt.subplots(1,2, figsize=(12,4))

# Plot 1: Time vs Dimension (Log Scale Y)
sns.lineplot(data=df_covered, x='d', y='Covered', hue='Method', style = 'Method', markers = True, ax=axes[0])
axes[0].set_ylabel('Coverage Probability')
axes[0].set_xlabel('Dimension (d)')
axes[0].grid(True, which="both", ls="--", alpha=0.5)
axes[0].legend(loc = 'lower left')
axes[0].axhline(y=1-alpha, color='black', linestyle='--', label=f'Nominal: {1-alpha}\%')
axes[0].set_ylim(0, 1)


# Plot 2: Mean Estimate vs Dimension
sns.lineplot(data=df_power, x='d', y='Power', hue='Method', style = 'Method', markers = True, ax=axes[1])
axes[1].set_ylabel('Power')
axes[1].set_xlabel('Dimension (d)')
axes[1].grid(True, which="both", ls="--", alpha=0.5)
axes[1].legend()
axes[1].get_legend().remove()


plt.tight_layout()
plt.savefig(f'figures/coverage_d{d_values}_metric{distance_metric}_B{B}_Bfrt{B_frt}_pval{p_val}_temperature{temperature}_eta{eta}_linear{linear}.pdf', bbox_inches='tight', dpi = 600)
# plt.show()