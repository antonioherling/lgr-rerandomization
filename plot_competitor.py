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
# d_values = [5,10,50,100,150,200,250]
d_values = [10,50,100,150,200,250]
# d_values = [2,3,4,5,6,7,8,9,10]  
B = 1000  # Number of repetitions per dimension
n = 500 # Total units
n1 = 250 # Treated units
p_val = 0.01 # Top 1% balance
n_jobs = -2  # Parallel cores
tau = 0.1 # True ATE
distance_metric = "mahalanobis" # "mahalanobis" or "euclidean"
temperature = 0.5  # LGR temperature
eta = 1.0  # LGR eta parameter
linear = True # linear DGP

file = f'results/competitor_d{d_values}_metric{distance_metric}_pval{p_val}_n{n}_temperature{temperature}_eta{eta}_linear{linear}.pkl'

try:
    df = pd.read_pickle(file)
    print(f"Loaded existing results from {file}")
except FileNotFoundError:
    print("No existing results found. Running simulations...")
    print("Run the file 'competitor_simulation.py' to generate the results.")
    exit()

# Reshape for Seaborn (Wide to Long)
time_cols = [c for c in df.columns if 'Time' in c]
iter_cols = [c for c in df.columns if 'Iter' in c]
bias_cols = [c for c in df.columns if 'Bias' in c]
dist_cols = [c for c in df.columns if 'Distance' in c]

df_time = pd.melt(df, id_vars=['d'], value_vars=time_cols, 
                  var_name='Method', value_name='Time')
df_iter = pd.melt(df, id_vars=['d'], value_vars=iter_cols, 
                  var_name='Method', value_name='Iterations')
df_bias = pd.melt(df, id_vars=['d'], value_vars=bias_cols, 
                 var_name='Method', value_name='Bias')
df_dist = pd.melt(df, id_vars=['d'], value_vars=dist_cols, 
                  var_name='Method', value_name='Mahalanobis_Distance')

# Clean names
df_time['Method'] = df_time['Method'].str.replace('_Time', '')
df_iter['Method'] = df_iter['Method'].str.replace('_Iter', '')
df_bias['Method'] = df_bias['Method'].str.replace('_Bias', '')
df_dist['Method'] = df_dist['Method'].str.replace('_Distance', '')



# # Remove CR from time plots (not rerandomization)
# df_time = df_time[df_time['Method'] != 'CR']


# ==========================================
# PLOTTING
# ==========================================
fig, axes = plt.subplots(2,2, figsize=(20, 10))

# Plot 1: Time vs Dimension (Log Scale Y)
sns.lineplot(data=df_time, x='d', y='Time', hue='Method', marker='o', ax=axes[0,0], lw=2)
axes[0,0].set_yscale('log')
axes[0,0].set_title('Computational Time vs. Dimension (Log Scale)')
axes[0,0].set_ylabel('Time (Seconds)')
axes[0,0].set_xlabel('Dimension (d)')
axes[0,0].grid(True, which="both", ls="--", alpha=0.5)

# Plot 2: Iterations vs Dimension (Log Scale Y)
sns.lineplot(data=df_iter, x='d', y='Iterations', hue='Method', marker='o', ax=axes[0,1], lw=2)
axes[0,1].set_yscale('log')
axes[0,1].set_title('Iterations vs. Dimension (Log Scale)')
axes[0,1].set_ylabel('Number of Iterations')
axes[0,1].set_xlabel('Dimension (d)')
axes[0,1].grid(True, which="both", ls="--", alpha=0.5)

# Plot 3: Estimate vs Dimension
sns.kdeplot(data=df_bias[df_bias['d'] == max(d_values)], x='Bias', hue='Method', fill = True, ax=axes[1,0])
# axes[1,0].axvline(tau, color='black', linestyle='--', label='True ATE')
axes[1,0].set_title(f'Bias vs. Dimension distribution at Highest Dimension (d = {max(d_values)})')
axes[1,0].set_xlabel(f'Bias')

# Plot 4: Mahalanobis Distance vs Dimension (Log Scale Y)
sns.kdeplot(data=df_dist[df_dist['d'] == max(d_values)], x='Mahalanobis_Distance', hue='Method', fill = True, ax=axes[1,1])
axes[1,1].set_title(f'Mahalanobis Distance distribution at Highest Dimension (d = {max(d_values)})')
axes[1,1].set_xlabel(f'Mahalanobis Distance')
axes[1,1].axvline(chi2.ppf(p_val, df=max(d_values)), color='black', linestyle='--', label='Balance Threshold (a)')
# sns.kdeplot(chi, ax=axes[1,1], color='black', linestyle='--', label='Chi-Squared Dist')
# axes[1,1].legend()


plt.tight_layout()
plt.savefig(f'competitor_d{d_values}_metric{distance_metric}_pval{p_val}_n{n}_temperature{temperature}_eta{eta}.pdf')
# plt.show()


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
sns.lineplot(data=df_time, x='d', y='Time', hue='Method', style = 'Method', markers = True, ax=axes[0])
axes[0].set_yscale('log')
# axes[0].set_title('Computational Time vs. Dimension (Log Scale)')
axes[0].set_ylabel('Time (Seconds)')
axes[0].set_xlabel('Dimension (d)')
axes[0].grid(True, which="both", ls="--", alpha=0.5)
if d_values == [2,3,4,5,6,7,8,9,10]:
    axes[0].legend(loc = 'lower left')
elif d_values == [10,50,100,150,200,250]:
    axes[0].legend(loc = 'lower left')

# Plot 2: Mean Estimate vs Dimension
sns.lineplot(data=df_bias, x='d', y='Bias', hue='Method', style = 'Method', markers = True, errorbar='sd', ax=axes[1])
# axes[1].axhline(0, color='black', linestyle='--', label='True ATE')
# axes[1].set_title(f'Mean Estimated ATE vs. Dimension')
axes[1].set_ylabel(f'Bias')
axes[1].set_xlabel('Dimension (d)')
# axes[1].grid(True, which="both", ls="--", alpha=0.5)
axes[1].legend()
axes[1].get_legend().remove()


plt.tight_layout()
plt.savefig(f'figures/competitor_d{d_values}_metric{distance_metric}_pval{p_val}_n{n}_temperature{temperature}_eta{eta}_linear{linear}.pdf', bbox_inches='tight', dpi = 600)
# plt.show()