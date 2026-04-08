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
d_values = [10,50,100,150,200,250]
# d_values = [2,3,4,5,6,7,8,9,10]  
B = 1000  # Number of repetitions per dimension
n = 500 # Total units
n1 = 250 # Treated units
p_val = 0.01 # Top 1% balance
n_jobs = -2  # Parallel cores
tau = 0.5 # True ATE
distance_metric = "mahalanobis" # "mahalanobis" or "euclidean"
temperatures = [0.01, 0.1, 0.5, 1.0, 10.0]  # LGR temperature
eta = 1.0 # LGR eta parameter
linear = True # linear DGP

files = [f'results/competitor_d{d_values}_metric{distance_metric}_pval{p_val}_n{n}_temperature{temperature}_eta{eta}_linear{linear}.pkl' for temperature in temperatures]

df = pd.DataFrame()
for file in files:
    print(f"Try to load results from {file}")
    try:
        df_file = pd.read_pickle(file)
        df_file['Temperature'] = file.split('_temperature')[1].split('.pkl')[0]
    except FileNotFoundError:
        print("No existing results found. Running simulations...")
        print("Run the file 'competitor_simulation.py' to generate the results.")
        exit()
    df = pd.concat([df, df_file], ignore_index=True)

# Reshape for Seaborn (Wide to Long)
time_cols = [c for c in df.columns if 'Time' in c]
iter_cols = [c for c in df.columns if 'Iter' in c]
bias_cols = [c for c in df.columns if 'Bias' in c]
dist_cols = [c for c in df.columns if 'Distance' in c]

df_time_temperatures = pd.melt(df, id_vars=['d', 'Temperature'], value_vars=time_cols, 
                  var_name='Method', value_name='Time')
df_iter_temperatures = pd.melt(df, id_vars=['d', 'Temperature'], value_vars=iter_cols, 
                  var_name='Method', value_name='Iterations')
df_bias_temperatures = pd.melt(df, id_vars=['d', 'Temperature'], value_vars=bias_cols, 
                 var_name='Method', value_name='Bias')
df_dist_temperatures = pd.melt(df, id_vars=['d', 'Temperature'], value_vars=dist_cols, 
                  var_name='Method', value_name='Mahalanobis_Distance')

df_time_temperatures['Method'] = df_time_temperatures['Method'].where( ~df_time_temperatures['Method'].eq('LGR_Time'), 'LGR_' + df_time_temperatures['Temperature'].astype(str) + '_Time' )
df_iter_temperatures['Method'] = df_iter_temperatures['Method'].where( ~df_iter_temperatures['Method'].eq('LGR_Iter'), 'LGR_' + df_iter_temperatures['Temperature'].astype(str) + '_Iter' )
df_bias_temperatures['Method'] = df_bias_temperatures['Method'].where( ~df_bias_temperatures['Method'].eq('LGR_Bias'), 'LGR_' + df_bias_temperatures['Temperature'].astype(str) + '_Bias' )
df_dist_temperatures['Method'] = df_dist_temperatures['Method'].where( ~df_dist_temperatures['Method'].eq('LGR_Distance'), 'LGR_' + df_dist_temperatures['Temperature'].astype(str) + '_Distance' )

for _df in [df_time_temperatures, df_iter_temperatures, df_bias_temperatures, df_dist_temperatures]:
    _df = _df.drop(columns='Temperature')

# Clean names
df_time_temperatures['Method'] = df_time_temperatures['Method'].str.replace('_Time', '')
df_iter_temperatures['Method'] = df_iter_temperatures['Method'].str.replace('_Iter', '')
df_bias_temperatures['Method'] = df_bias_temperatures['Method'].str.replace('_Bias', '')
df_dist_temperatures['Method'] = df_dist_temperatures['Method'].str.replace('_Distance', '')


# Exclude methods
EXCLUDE_METHODS = ['CR', 'ARR']
df_time_temperatures = df_time_temperatures[~df_time_temperatures['Method'].isin(EXCLUDE_METHODS)]
df_iter_temperatures = df_iter_temperatures[~df_iter_temperatures['Method'].isin(EXCLUDE_METHODS)]
df_bias_temperatures = df_bias_temperatures[~df_bias_temperatures['Method'].isin(EXCLUDE_METHODS)]
df_dist_temperatures = df_dist_temperatures[~df_dist_temperatures['Method'].isin(EXCLUDE_METHODS)]

# Rename LGR methods for clarity
def relabel_method(m):
    if m.startswith('LGR_'):
        return rf'$\delta = {m.split("_")[1]}$'
    return m

for _df in [df_time_temperatures, df_iter_temperatures, df_bias_temperatures, df_dist_temperatures]:
    _df['Method'] = _df['Method'].apply(relabel_method)


# ============ VARYING ETA PARAMETER ==============


# ==========================================
# CONFIGURATION
# ==========================================
# Dimensions to test (The X-axis)
# We stop Rejection sampling early because it will hang forever at high d
d_values = [10,50,100,150,200,250]
# d_values = [2,3,4,5,6,7,8,9,10]  
B = 1000  # Number of repetitions per dimension
n = 500 # Total units
n1 = 250 # Treated units
p_val = 0.01 # Top 1% balance
n_jobs = -2  # Parallel cores
tau = 0.5 # True ATE
distance_metric = "mahalanobis" # "mahalanobis" or "euclidean"
temperature = 0.5  # LGR temperature
etas = [0.01, 0.1, 1.0, 10.0] # LGR eta parameter
linear = True # linear DGP

files = [f'results/competitor_d{d_values}_metric{distance_metric}_pval{p_val}_n{n}_temperature{temperature}_eta{eta}_linear{linear}.pkl' for eta in etas]

df = pd.DataFrame()
for file in files:
    print(f"Try to load results from {file}")
    try:
        df_file = pd.read_pickle(file)
        df_file['Eta'] = file.split('_eta')[1].split('.pkl')[0]
    except FileNotFoundError:
        print("No existing results found.")
        print("Run the file 'competitor_simulation.py' to generate the results.")
        exit()
    df = pd.concat([df, df_file], ignore_index=True)

# Reshape for Seaborn (Wide to Long)
time_cols = [c for c in df.columns if 'Time' in c]
iter_cols = [c for c in df.columns if 'Iter' in c]
bias_cols = [c for c in df.columns if 'Bias' in c]
dist_cols = [c for c in df.columns if 'Distance' in c]

df_time_etas = pd.melt(df, id_vars=['d', 'Eta'], value_vars=time_cols, 
                  var_name='Method', value_name='Time')
df_iter_etas = pd.melt(df, id_vars=['d', 'Eta'], value_vars=iter_cols, 
                  var_name='Method', value_name='Iterations')
df_bias_etas = pd.melt(df, id_vars=['d', 'Eta'], value_vars=bias_cols, 
                 var_name='Method', value_name='Bias')
df_dist_etas = pd.melt(df, id_vars=['d', 'Eta'], value_vars=dist_cols, 
                  var_name='Method', value_name='Mahalanobis_Distance')

df_time_etas['Method'] = df_time_etas['Method'].where( ~df_time_etas['Method'].eq('LGR_Time'), 'LGR_' + df_time_etas['Eta'].astype(str) + '_Time' )
df_iter_etas['Method'] = df_iter_etas['Method'].where( ~df_iter_etas['Method'].eq('LGR_Iter'), 'LGR_' + df_iter_etas['Eta'].astype(str) + '_Iter' )
df_bias_etas['Method'] = df_bias_etas['Method'].where( ~df_bias_etas['Method'].eq('LGR_Bias'), 'LGR_' + df_bias_etas['Eta'].astype(str) + '_Bias' )
df_dist_etas['Method'] = df_dist_etas['Method'].where( ~df_dist_etas['Method'].eq('LGR_Distance'), 'LGR_' + df_dist_etas['Eta'].astype(str) + '_Distance' )

for _df in [df_time_etas, df_iter_etas, df_bias_etas, df_dist_etas]:
    _df = _df.drop(columns='Eta')

# Clean names
df_time_etas['Method'] = df_time_etas['Method'].str.replace('_Time', '')
df_iter_etas['Method'] = df_iter_etas['Method'].str.replace('_Iter', '')
df_bias_etas['Method'] = df_bias_etas['Method'].str.replace('_Bias', '')
df_dist_etas['Method'] = df_dist_etas['Method'].str.replace('_Distance', '')


# Exclude methods
EXCLUDE_METHODS = ['CR', 'ARR']

df_time_etas = df_time_etas[~df_time_etas['Method'].isin(EXCLUDE_METHODS)]
df_iter_etas = df_iter_etas[~df_iter_etas['Method'].isin(EXCLUDE_METHODS)]
df_bias_etas = df_bias_etas[~df_bias_etas['Method'].isin(EXCLUDE_METHODS)]
df_dist_etas = df_dist_etas[~df_dist_etas['Method'].isin(EXCLUDE_METHODS)]

# Rename LGR methods for clarity
def relabel_method(m):
    if m.startswith('LGR_'):
        return rf'$\eta = {m.split("_")[1]}$'
    return m

for _df in [df_time_etas, df_iter_etas, df_bias_etas, df_dist_etas]:
    _df['Method'] = _df['Method'].apply(relabel_method)


# ==========================================
# PLOTTING
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

fig, axes = plt.subplots(2,2, figsize=(12,8))

# Plot 1: Time vs Dimension (Log Scale Y) for temperatures
sns.lineplot(data=df_time_temperatures, x='d', y='Time', hue='Method', style = 'Method', markers = True, ax=axes[0,0], lw=2)
axes[0,0].set_yscale('log')
# axes[0].set_title('Computational Time vs. Dimension (Log Scale)')
axes[0,0].set_ylabel('Time (Seconds)')
axes[0,0].set_xlabel('')
axes[0,0].grid(True, which="both", ls="--", alpha=0.5)
axes[0,0].legend(loc='upper left')

# Plot 2: Time vs Dimension (Log Scale Y) for etas
sns.lineplot(data=df_time_etas, x='d', y='Time', hue='Method', style = 'Method', markers = True, ax=axes[0,1], lw=2)
axes[0,1].set_yscale('log')
# axes[1].set_title('Mean Estimate vs. Dimension')
axes[0,1].set_ylabel('')
axes[0,1].set_xlabel('')
axes[0,1].grid(True, which="both", ls="--", alpha=0.5)
axes[0,1].legend(loc='upper left')


# Plot 3: Bias vs Dimension for temperatures
sns.lineplot(data=df_bias_temperatures, x='d', y='Bias', hue='Method', style = 'Method', markers = True, errorbar='sd', ax=axes[1,0], lw=2)
# axes[1].axhline(0, color='black', linestyle='--', label='True ATE')
# axes[1].set_title(f'Mean Estimated ATE vs. Dimension')
axes[1,0].set_ylabel(f'Bias')
axes[1,0].set_xlabel('Dimension (d)')
# axes[1].grid(True, which="both", ls="--", alpha=0.5)
axes[1,0].legend(loc='upper left')
# axes[1,0].get_legend().remove()

# Plot 4: Bias vs Dimension for etas
sns.lineplot(data=df_bias_etas, x='d', y='Bias', hue='Method', style = 'Method', markers = True, errorbar='sd', ax=axes[1,1], lw=2)
# axes[3].axhline(0, color='black', linestyle='--', label='True ATE')
# axes[3].set_title(f'Mean Estimated ATE vs. Dimension')
axes[1,1].set_ylabel(f'')
axes[1,1].set_xlabel('Dimension (d)')
# axes[3].grid(True, which="both", ls="--", alpha=0.5)
axes[1,1].legend(loc='upper left')
# axes[1,1].get_legend().remove()


plt.tight_layout()
plt.savefig(f'figures/sensitivity_d{d_values}_metric{distance_metric}_pval{p_val}_n{n}_temperatures{temperatures}_etas{etas}_linear{linear}.pdf', bbox_inches='tight', dpi = 600)
# plt.show()