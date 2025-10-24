#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.mixed_linear_model import MixedLM
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
)

from scipy.stats import linregress

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

notebook_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(notebook_dir, os.pardir))
os.chdir(parent_dir)
print("Working directory set to:", os.getcwd())


# In[2]:


os.chdir('datasets')
print(os.getcwd())

df = pd.read_csv('holistic_wheat_merged_multi.csv')
df.sort_values(by='Date', inplace = True)

df_y = pd.read_csv('holistic_wheat_yield_merged.csv')

df = pd.merge(df, df_y[['Loc', 'PN', 'Treatment', 'Code', 'Yield (kg/ha)']], 
               on=['Loc', 'PN', 'Treatment', 'Code'], how='left')

os.chdir('..')
os.chdir('results')

stability_data = pd.read_csv('yield_stability_groups.csv')

df = pd.merge(df, stability_data[['Code', 'Stability', 'Stability group']],
                        on=['Code'], how ='left')

os.chdir('..')
os.chdir('datasets\\meteo')

ggd = pd.read_csv("GDD_dates_all_locations.csv")

df = pd.merge(df,ggd[['Date','GDD']], on=['Date'], how='left')

ETo = pd.read_csv("ETo_prec_all_locations.csv")

df = pd.merge(df,ETo[['Date','Treatment','ETo (mm)' , 'water_input_cumulative']], on=['Date', 'Treatment'], how='left')

# For posterior model fitting: delete * and / symbols that give errors in LMM
# Remove '*' from all column names
df.columns = df.columns.str.replace(r'\*', '', regex=True)
df.columns = df.columns.str.replace(r'\/', '_', regex=True)


df['Tveg_mean_UAV'] = (
    (df['Tveg_mean_UAV'] - df.groupby(['Date', 'Loc'])['Tveg_mean_UAV'].transform('mean')) /
    df.groupby(['Date', 'Loc'])['Tveg_mean_UAV'].transform('std')
)


# Modelling datasets
df_a = df[(df['Date'] == '2024-04-22') | (df['Date'] == '2024-05-14')] # Aranjuez
df_v = df[(df['Date'] == '2024-05-29') | (df['Date'] == '2024-06-12')] # Valladolid
df_an = df[(df['Date'] == '2024-04-22') | (df['Date'] == '2024-05-29')] # Anthesis A-V
df_gf = df[(df['Date'] == '2024-05-14') | (df['Date'] == '2024-06-12')] # Grain Filling A-V


# In[3]:


# Variable selection per modeling dataset
id_columns = df.columns[0:11].to_list() 
yield_column = df.columns[49:50].to_list() 
stability_column = df.columns[50:51].to_list() 
meteo_column = df.columns[51:].to_list() 
vi_all =  df.columns[11:48].to_list() 

df_a = df_a[id_columns + meteo_column + vi_all + yield_column + stability_column]
df_v = df_v[id_columns + meteo_column + vi_all + yield_column + stability_column]
df_an = df_an[id_columns + meteo_column + vi_all + yield_column + stability_column]
df_gf = df_gf[id_columns + meteo_column + vi_all + yield_column + stability_column]


# In[4]:


def calculate_vi_differences(df, vi_columns):
    df['Date'] = pd.to_datetime(df['Date'])
    
    df = df.sort_values(by=['PN', 'Treatment', 'Date'])
    
    # Calculate the difference in GDD within each PN-Treatment group
    df['GDD_diff'] = df.groupby(['PN', 'Treatment'])['GDD'].diff()

    # Calculate the difference in vegetation indices and divide by GDD difference
    df[vi_columns] = df.groupby(['PN', 'Treatment'])[vi_columns].diff().div(df['GDD_diff'], axis=0)
    
    df_diff = df.dropna(subset=vi_columns)
    
    return df_diff

df_a_diff = calculate_vi_differences(df_a, vi_all)
df_v_diff = calculate_vi_differences(df_v, vi_all)

df_diff = pd.concat([df_a_diff, df_v_diff], ignore_index=True)
print(df_diff)


# In[5]:


def normalize_indexes(df, index_columns):
    data = df.copy()
    for index in index_columns: 
        if index != 'Tveg_mean_UAV':
            data[index] = (
            (data[index] - data[index].mean()) /
            data[index].std()
            )
    return data

df_diff_n = normalize_indexes(df_diff, vi_all)
df_an_n = normalize_indexes(df_an, vi_all)
df_gf_n = normalize_indexes(df_gf, vi_all)
print(df_diff_n)


# In[6]:


def calculate_index_stability(df, veg_index_columns):
    df = df.copy()
        
    # Define environment and block identifiers
    df['Loc_Treatment'] = df['Loc'] + "_" + df['Treatment']
    df['Block_ID'] = df['Loc_Treatment'] + "_" + df['B'].astype(str)
    
    df['Code'] = df['Code'].astype('category')
    df['Loc_Treatment'] = df['Loc_Treatment'].astype('category')
    df['Block_ID'] = df['Block_ID'].astype('category')

    index_stab_df = pd.DataFrame(columns=['Index', 'Code', 'I_Stability', 'Intercept', 'Residual_Std'])

    for index in veg_index_columns:
        # Environmental index = mean VI per environment
        df['Environmental_Index'] = df.groupby('Loc_Treatment', observed=False)[index].transform('mean')

        # Fit mixed model (block nested in environment as random effect)
        formula = f"{index} ~ Environmental_Index * Code"
        model = smf.mixedlm(formula=formula, data=df, groups=df["Block_ID"])
        results = model.fit(reml=True)

        coeffs = results.params
        main_effect_env = coeffs["Environmental_Index"]  # β1

        # Extract interaction terms (β3) for each genotype
        interaction_terms = coeffs.filter(like="Environmental_Index:Code")
        interaction_df = interaction_terms.reset_index()
        interaction_df.columns = ['Term', 'I_Stability']
        interaction_df['Code'] = interaction_df['Term'].str.extract(r'Code\[(.+?)\]')

        # Adjust slopes
        interaction_df['I_Stability'] += main_effect_env

        # Add reference genotype (baseline slope)
        reference_row = pd.DataFrame({
            'Code': [df['Code'].cat.categories[0]],
            'I_Stability': [main_effect_env]
        })
        interaction_df = pd.concat([interaction_df, reference_row], ignore_index=True)

        # Extract intercepts
        reference_intercept = coeffs["Intercept"]
        genotype_effects = coeffs.filter(like="Code").drop(labels=interaction_terms.index, errors='ignore')
        intercept_df = genotype_effects.reset_index()
        intercept_df.columns = ['Term', 'Beta2']
        intercept_df['Code'] = intercept_df['Term'].str.extract(r'Code\[(.+?)\]')
        intercept_df['Intercept'] = reference_intercept + intercept_df['Beta2']

        # Add reference genotype intercept
        ref_intercept_row = pd.DataFrame({
            'Code': [df['Code'].cat.categories[0]],
            'Intercept': [reference_intercept]
        })
        intercept_df = pd.concat([intercept_df[['Code', 'Intercept']], ref_intercept_row], ignore_index=True)

 
        stability_df = pd.merge(interaction_df[['Code', 'I_Stability']], intercept_df, on='Code', how='outer')
        stability_df['Index'] = index
        stability_df['Residual_Std'] = np.sqrt(results.scale)
        stability_df['Code'] = stability_df['Code'].str.replace(r'^T\.', '', regex=True)

        index_stab_df = pd.concat([index_stab_df, stability_df], ignore_index=True)


    df_wide = index_stab_df.pivot(index='Code', columns='Index', values='I_Stability')
    if 'Stability' in df.columns:
        df_wide = df_wide.merge(df[['Code', 'Stability']].drop_duplicates(), on='Code', how='left')

    return index_stab_df, df_wide

stability_diff, wide_diff = calculate_index_stability(df_diff, vi_all)
stability_an, wide_an = calculate_index_stability(df_an, vi_all)
stability_gf, wide_gf = calculate_index_stability(df_gf, vi_all)


# In[13]:


def correlate_index_stability(wide_df, dataset_name="Dataset"):
    corr_df = wide_df.dropna()


    correlations = corr_df.corr(numeric_only=True)['Stability'].drop('Stability')

    corr_table = correlations.reset_index()
    corr_table.columns = ['Index', 'Correlation_with_Stability']
    corr_table['Dataset'] = dataset_name

    return corr_table


corr_diff = correlate_index_stability(wide_diff, "Index_Difference")
corr_an = correlate_index_stability(wide_an, "Anthesis")
corr_gf = correlate_index_stability(wide_gf, "Grain_Filling")


correlation_summary = pd.concat([corr_an, corr_gf, corr_diff], ignore_index=True)


correlation_summary = correlation_summary[['Dataset', 'Index', 'Correlation_with_Stability']] \
    .assign(abs_corr=lambda x: x['Correlation_with_Stability'].abs()) \
    .sort_values('abs_corr', ascending=False) \
    .drop(columns='abs_corr')

print(correlation_summary.head(38))

os.chdir('..')
os.chdir('..\\results')

correlation_summary.to_csv("VI_stability_stability_correlations.csv", index=False)


# In[13]:


def feature_selection_rf_loo(df, veg_index_columns, target, min_features=2, max_features=6, direction='backward'):
    """
    Perform sequential feature selection using Leave-One-Out (LOO) cross-validation 
    and include the genotype name (under 'Code') for each holdout sample in the results.
    """
    df = df.dropna()  

    X = df[veg_index_columns]
    y = df[target]
    genotypes = df['Code']  

    common_index = X.index.intersection(y.index).intersection(genotypes.index)
    X = X.loc[common_index]
    y = y.loc[common_index]
    genotypes = genotypes.loc[common_index]

    # Initialize Random Forest
    rf = RandomForestRegressor(random_state=32, n_estimators=100)

    # Sequential Feature Selector
    sfs = SFS(
        rf,
        k_features=(min_features, max_features),
        forward=(direction == 'forward'),
        floating=False,
        scoring='neg_mean_absolute_error',
        cv=LeaveOneOut(),
        n_jobs=-1
    )

    print(f"Starting sequential feature selection for target '{target}'...")
    try:
        sfs.fit(X, y)
        best_features = list(X.columns[list(sfs.k_feature_idx_)])
    except Exception as e:
        print(f"Feature selection failed: {e}")
        return None

    print(f"Selected features: {best_features}")

    loo_results = []

    # Leave-One-Out Cross-Validation
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx][best_features], X.iloc[test_idx][best_features]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        genotype_test = genotypes.iloc[test_idx].values[0]  # Genotype name for the holdout sample

        rf.fit(X_train, y_train)

        # Predict for the holdout sample
        y_pred = rf.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  

        loo_results.append({
            'Genotype (Code)': genotype_test,
            'Observed': y_test.values[0],
            'Predicted': y_pred[0],
            'RMSE': rmse,
            'MAE': mae,
            'MAPE (%)': mape
        })

    results_df = pd.DataFrame(loo_results)
    return best_features, results_df

best_features_diff, loo_results_diff = feature_selection_rf_loo(wide_diff, vi_all, 'Stability')

print("Best Features Selected DIFF:", best_features_diff)
print("\nLOO Results:\n", loo_results_diff)

best_features_an, loo_results_an = feature_selection_rf_loo(wide_an, vi_all, 'Stability')

print("Best Features Selected Anthesis:", best_features_an)
print("\nLOO Results:\n", loo_results_an)

best_features_gf, loo_results_gf = feature_selection_rf_loo(wide_gf, vi_all, 'Stability')

print("Best Features Selected Grain Filling:", best_features_gf)
print("\nLOO Results:\n", loo_results_gf)


# In[17]:


def summarize_results(results_df):
    summary = results_df[['RMSE', 'MAE', 'MAPE (%)']].agg(['mean', 'std']).T
    summary.columns = ['Mean', 'Std Dev']
    return summary

summary_diff = summarize_results(loo_results_diff)
summary_an = summarize_results(loo_results_an)
summary_gf = summarize_results(loo_results_gf)

print("Summary for DIFF:")
print(summary_diff)

print("Summary for Anthesis:")
print(summary_an)

print("Summary for Grain Filling:")
print(summary_gf)


# In[19]:


os.getwd()


# In[21]:


os.chdir('..')
os.chdir('..\\results\\feature_selection')


# In[23]:


def process_stability_results(results_df, best_features, filename):
    results = results_df.copy()
    results = results.rename(columns={'MAPE (%)': 'MAPE'})
    results = results.rename(columns={'Genotype (Code)': 'Code'})
    results = pd.merge(results, stability_data[['Code', 'Stability group']], on='Code', how='left')
    results = results.rename(columns={'Stability group': 'Stability group observed'})

    # Compute quartiles
    upper_quartile = results["Predicted"].quantile(0.75)
    lower_quartile = results["Predicted"].quantile(0.25)

    def classify_stability(value):
        if value >= upper_quartile:
            return "Low stability"
        elif value <= lower_quartile:
            return "High stability"
        else:
            return "Intermediate stability"

    results["Stability group predicted"] = results["Predicted"].apply(classify_stability)

    best_features_str = ", ".join(best_features)
    results['Best Features'] = best_features_str

    results.to_csv(filename, index=False)

    return results

pred_results_diff = process_stability_results(loo_results_diff, best_features_diff, "stability_loo_diff.csv")
pred_results_an = process_stability_results(loo_results_an, best_features_an, "stability_loo_an.csv")
pred_results_gf = process_stability_results(loo_results_gf, best_features_gf, "stability_loo_gf.csv")

pred_results_diff.head()


# In[35]:


# os.chdir('results\\feature_selection')
os.chdir('..')
os.chdir('..\\results\\feature_selection')


# In[37]:


pred_results_diff = pd.read_csv("stability_loo_diff.csv")
pred_results_an = pd.read_csv("stability_loo_an.csv")
pred_results_gf = pd.read_csv("stability_loo_gf.csv")

 #['v_UAV', 'CSI_UAV', 
VI_selected_an = ['GGA_UAV', 'NGRDI_UAV', 'NDVI_UAV', 'SAVI_UAV', 'Tveg_mean_UAV', 'u*_ground']
VI_selected_gf = ['Hue_UAV', 'Saturation_UAV', 'Tveg_mean_UAV', 'NDVI_ground', 'v*_ground']
VI_selected_diff = ['Saturation_UAV', 'NGRDIveg_UAV', 'NDVI_UAV', 'Lightness_ground', 'a*_ground', 'TGI_ground']


def plot_stability_comparison(ax, df, title):
    sns.scatterplot(x=df['Observed'], y=df['Predicted'], ax=ax, color='royalblue', alpha=0.7)
    
    ax.plot([df['Observed'].min(), df['Observed'].max()],
            [df['Observed'].min(), df['Observed'].max()],
            color='black', linestyle='--')

    # Regression stats
    slope, intercept, r_value, p_value, std_err = linregress(df['Observed'], df['Predicted'])
    r_squared = r_value ** 2

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Observed Stability', fontsize=16)
    ax.set_ylabel('Predicted Stability', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    # Error metrics
    metrics = ['RMSE', 'MAE', 'MAPE']
    mean_std_values = {metric: (df[metric].mean(), df[metric].std()) for metric in metrics if metric in df.columns}
    error_text = '\n'.join([f"{metric}: {mean:.2f} (±{std:.2f})" for metric, (mean, std) in mean_std_values.items()])
    error_text += f"\nR²: {r_squared:.3f}"
    error_text += f"\np-value: {p_value:.3f}"

    ax.text(0.05, 0.95, error_text, transform=ax.transAxes, fontsize=16, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))

fig, axes = plt.subplots(3, 1, figsize=(8, 18))
plt.subplots_adjust(hspace=0.0)

plot_stability_comparison(axes[0], pred_results_an, 'Anthesis')
plot_stability_comparison(axes[1], pred_results_gf, 'Grain filling')
plot_stability_comparison(axes[2], pred_results_diff, 'Index-difference')

# Add selected indices below each subplot
vi_lists = [VI_selected_an, VI_selected_gf, VI_selected_diff]

for ax, VI_selected in zip(axes, vi_lists):
    half = len(VI_selected) // 2
    vi_text_1 = ', '.join(VI_selected[:half])
    vi_text_2 = ', '.join(VI_selected[half:])
    ax.text(0.5, -0.15, f"Selected features: {vi_text_1}", ha='center', va='top',
            fontsize=16, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, -0.22, vi_text_2, ha='center', va='top',
            fontsize=16, fontweight='bold', transform=ax.transAxes)

plt.tight_layout()

os.chdir('..')
os.chdir('..\\results\\figures\\feature_selection')
plt.savefig("stability_prediction_results_onecol.png", dpi=300, bbox_inches='tight')

plt.show()

