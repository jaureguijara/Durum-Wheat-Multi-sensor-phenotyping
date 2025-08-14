#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene, ttest_ind, probplot, f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings("ignore")


# In[2]:


os.chdir('..')
os.chdir('results')


yield_groups = pd.read_csv("yield_groups_based_on_emms.csv")
stability_groups = pd.read_csv("yield_stability_groups.csv")

observed_groups = pd.merge(yield_groups, stability_groups[['Code', 'Stability', 'Stability group']], on ='Code')

observed_groups['Selection'] = np.where(
    (observed_groups['Yield group'] == 'Low yield') | (observed_groups['Stability group'] == 'Low stability'),
    'discard',
    'keep'
)
print(observed_groups.head())

# Count the number of genotypes in each Selection category
selection_counts = observed_groups['Selection'].value_counts()

print(selection_counts)

observed_groups.to_csv("selection_groups.csv", index=False)


# In[3]:


# Filter discarded genotypes (where 'Selection' is 'discard') 
discarded_observed = observed_groups[observed_groups['Selection'] == 'discard']['Code'].tolist()
print("Discarded genotypes:")
print(discarded_observed)


# In[4]:


## Plot Yield, yield stability and genotype selection groups  against environmental index

os.chdir('..')
os.chdir('datasets')

df = pd.read_csv('holistic_wheat_yield_merged.csv')
df['Code'] = df['Code'].astype('category')
df['Loc_Treatment'] = df['Loc'] + "_" + df['Treatment']
df['Loc_Treatment'] = df['Loc_Treatment'].astype('category')

df = df.rename(columns={'Yield (kg/ha)': 'Yield_kg_ha'})

df_m = pd.merge(df, observed_groups[['Yield group', 'Stability', 'Stability group', 'Selection', 'Code']], on='Code', how='left')

df_test = df_m.dropna(subset=['Yield_kg_ha', 'Selection'])

group_keep = df_test[df_test['Selection'] == 'keep']['Yield_kg_ha']
group_discard = df_test[df_test['Selection'] == 'discard']['Yield_kg_ha']


# In[5]:


df_clean = df_m.dropna(subset=['Yield_kg_ha'])

df_clean = df_clean.rename(columns={'Yield group': 'Yield_group'})
df_clean = df_clean.rename(columns={'Stability group': 'Stability_group'})
df_clean = df_clean.rename(columns={'Selection': 'Selection_group'})

# Environmental index (mean yield per environment)
ei = df_clean.groupby('Loc_Treatment')['Yield_kg_ha'].mean().reset_index().rename(columns={'Yield_kg_ha': 'Environmental_Index'})
df_clean = df_clean.merge(ei, on='Loc_Treatment', how='left')
df_clean.head()


# In[6]:


os.chdir('..')
os.chdir('results\\figures')


# In[25]:


def lmm_analysis_combined_plot(df, group_cols, palette='colorblind'):

    locs = df['Loc_Treatment'].unique()
    n_groups = len(group_cols)

    fixed_colors = ['#009E73',  
                    '#E69F00',  
                    '#0072B2']  

    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 6), sharey=True)
    if n_groups == 1:
        axes = [axes]

    for i, (ax, group_col) in enumerate(zip(axes, group_cols)):
        print(f"\n=== Analyzing group: {group_col} ===")

        formula = f"Yield_kg_ha ~ Environmental_Index * {group_col}"
        lmm = MixedLM.from_formula(formula, groups=df["Loc_Treatment"], data=df)
        result = lmm.fit()
        print(result.summary())

        unique_groups = sorted(df[group_col].dropna().unique())

        if group_col == 'Selection_group':
            color_map = {
                'keep': '#0072B2',    
                'discard': '#E69F00' 
            }
        else:
            # Assign fixed colors in alphabetical order, cycling if more than 3 groups
            assigned_colors = fixed_colors * ((len(unique_groups) // len(fixed_colors)) + 1)
            color_map = dict(zip(unique_groups, assigned_colors[:len(unique_groups)]))

        env_vals = np.linspace(df['Environmental_Index'].min(), df['Environmental_Index'].max(), 100)
        for grp in unique_groups:
            pred_df = pd.DataFrame({
                'Environmental_Index': env_vals,
                group_col: [grp]*len(env_vals)
            })
            preds = result.predict(pred_df)
            ax.plot(env_vals, preds, label=f"{grp}", color=color_map[grp])

        stats_df = df.groupby(['Loc_Treatment', group_col])['Yield_kg_ha'].agg(
            mean='mean', std='std', count='count').reset_index()
        stats_df['se'] = stats_df['std'] / np.sqrt(stats_df['count'])
        env_index_map = df.groupby('Loc_Treatment')['Environmental_Index'].first().to_dict()
        
        for _, row in stats_df.iterrows():
            x_pos = env_index_map[row['Loc_Treatment']]
            ax.errorbar(x_pos, row['mean'], yerr=row['se'], fmt='o',
                        color=color_map[row[group_col]], capsize=5)

        for env in locs:
            sub_env = df[df['Loc_Treatment'] == env]
            groups_present = sub_env[group_col].dropna().unique()
        
            if len(groups_present) >= 2:
                groups_data = [g['Yield_kg_ha'].values for _, g in sub_env.groupby(group_col) if len(g) > 1]
                if len(groups_data) >= 2:
                    f_stat, p = f_oneway(*groups_data)
                    print(f\n ANOVA for {group_col} in {env}:" f_oneway(*groups_data))
                    
                    if p < 0.05:
                        # Mark significance with a star
                        env_stats = stats_df[stats_df['Loc_Treatment'] == env]
                        max_row = env_stats.loc[(env_stats['mean'] + env_stats['se']).idxmax()]
                        x_pos = env_index_map[env]
                        y_star = max_row['mean'] + max_row['se'] + max_row['se'] * 0.5
                        ax.text(x_pos, y_star*0.985, '*', ha='center', va='bottom', fontsize=18, color='red')
                        
        
                        # ---- Tukey HSD ----
                        tukey = pairwise_tukeyhsd(endog=sub_env['Yield_kg_ha'],
                                                  groups=sub_env[group_col],
                                                  alpha=0.05)

                        print(f"\nTukey HSD for {group_col} in {env}:\n")
                        print(tukey.summary())
        
        group_col_title = group_col.replace('_', ' ')

        ax.set_title(f'{group_col_title}', fontsize=20)
        # Remove individual xlabels to add a common one later
        # ax.set_xlabel('Environmental Index (Mean Yield per Environment)', fontsize=13)

        if i == 0:
            ax.set_ylabel('Yield (kg/ha)', fontsize=18)
        else:
            ax.set_ylabel('')
            ax.yaxis.set_tick_params(labelleft=True)

        ax.grid(False)
        ax.legend(fontsize=16, title_fontsize=18, loc='upper left')
        ax.tick_params(axis='x', labelsize=16)  
        ax.tick_params(axis='y', labelsize=16) 

    # Common xlabel
    fig.text(0.5, 0.04, 'Environmental Index (Mean Yield per Environment)', ha='center', fontsize=18)

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig('yield_stability_selection_lmm_plot.png', dpi=300)
    plt.show()


lmm_analysis_combined_plot(df_clean, ['Yield_group', 'Stability_group', 'Selection_group'])

