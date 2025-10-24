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

import warnings
warnings.filterwarnings("ignore")


# In[2]:


os.chdir('..')
os.chdir('results')


yield_groups = pd.read_csv("yield_groups_based_on_emms.csv")
stability_groups = pd.read_csv("yield_stability_groups.csv")

observed_groups = pd.merge(yield_groups, stability_groups[['Code', 'Stability', 'Stability group']], on ='Code')

observed_groups['Selection'] = np.where(
    (observed_groups['Yield_group'] == 'Low yield') | (observed_groups['Stability group'] == 'Low stability')
     #| (observed_groups['Stability group'] == 'High stability') # only when slopes =1 are desired
    , 
    'discard',
    'keep'
)


observed_groups = observed_groups.rename(columns={'Yield_group': 'Yield group'})
print(observed_groups.head())
selection_counts = observed_groups['Selection'].value_counts()

print(selection_counts)

#observed_groups.to_csv("selection_groups_alternative.csv", index=False)

# Grouped mean ± SD by Selection
grouped_stats = observed_groups.groupby('Selection').agg(
    mean_EMM=('emmean', 'mean'),
    sd_EMM=('emmean', 'std'),
    mean_Stability=('Stability', 'mean'),
    sd_Stability=('Stability', 'std'),
    n=('Code', 'count')
).reset_index()


grouped_stats['se_EMM'] = grouped_stats['sd_EMM'] / np.sqrt(grouped_stats['n'])
grouped_stats['se_Stability'] = grouped_stats['sd_Stability'] / np.sqrt(grouped_stats['n'])

print("\nMean ± SD by Selection group:")
for _, row in grouped_stats.iterrows():
    print(f"{row['Selection']:<10} | "
          f"Yield: {row['mean_EMM']:.2f} ± {row['sd_EMM']:.2f} | "
          f"Stability: {row['mean_Stability']:.3f} ± {row['sd_Stability']:.3f} | "
          f"n={row['n']}")


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
df_clean['Loc_Treatment_Block'] = df_clean['Loc_Treatment'].astype(str) + "_B" + df['B'].astype(str)
df_clean.head()


# In[6]:


os.chdir('..')
os.chdir('results\\figures')


# In[7]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, f, chi2
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def fw_mixedlm_analysis_combined_plot(df, group_cols, palette='colorblind'):
    # Create unique block ID nested within environment
    if 'Loc_Treatment_Block' not in df.columns:
        df['Loc_Treatment_Block'] = df['Loc_Treatment'].astype(str) + "_B" + df['Block'].astype(str)

    locs = df['Loc_Treatment'].unique()
    n_groups = len(group_cols)
    fixed_colors = ['#009E73', '#E69F00', '#0072B2']

    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 6), sharey=True)
    if n_groups == 1:
        axes = [axes]

    fw_results = []

    for i, (ax, group_col) in enumerate(zip(axes, group_cols)):
        print(f"\n=== Finlay–Wilkinson mixed model analysis for: {group_col} ===")

        unique_groups = sorted(df[group_col].dropna().unique())

        if group_col == 'Selection_group':
            color_map = {'keep': '#0072B2', 'discard': '#E69F00'}
        else:
            assigned_colors = fixed_colors * ((len(unique_groups) // len(fixed_colors)) + 1)
            color_map = dict(zip(unique_groups, assigned_colors[:len(unique_groups)]))

        env_vals = np.linspace(df['Environmental_Index'].min(), df['Environmental_Index'].max(), 100)

        for grp in unique_groups:
            sub_df = df[df[group_col] == grp]

            model = smf.mixedlm("Yield_kg_ha ~ Environmental_Index", 
                                data=sub_df, 
                                groups=sub_df["Loc_Treatment_Block"])
            result = model.fit(reml=True)

            intercept = result.params['Intercept']
            intercept_se = result.bse['Intercept']
            intercept_t = result.tvalues['Intercept']
            intercept_p = result.pvalues['Intercept']

            slope = result.params['Environmental_Index']
            slope_se = result.bse['Environmental_Index']
            slope_t = result.tvalues['Environmental_Index']
            slope_p = result.pvalues['Environmental_Index']

            model_no_random = smf.ols("Yield_kg_ha ~ Environmental_Index", data=sub_df).fit()
            llf_full = result.llf
            llf_null = model_no_random.llf
            LRT = 2 * (llf_full - llf_null)
            p_block = chi2.sf(LRT, df=1)

            y = sub_df['Yield_kg_ha']
            y_pred = result.fittedvalues
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            n = len(y)
            k = 1
            f_stat = (r2 / k) / ((1 - r2) / (n - k - 1)) if n > k + 1 else np.nan
            p_r2 = 1 - f.cdf(f_stat, k, n - k - 1) if n > k + 1 else np.nan

            print(f"\nGroup: {grp}")
            print(f"  Intercept: {intercept:.2f} ± {intercept_se:.2f}, t={intercept_t:.2f}, p={intercept_p:.4f}")
            print(f"  Slope: {slope:.3f} ± {slope_se:.3f}, t={slope_t:.2f}, p={slope_p:.4f}")
            print(f"  R² = {r2:.3f} (p={p_r2:.4f})")
            print(f"  Block variance = {result.cov_re.iloc[0,0]:.4f} (LRT p={p_block:.4f})")

            fw_results.append({
                'Group_Column': group_col,
                'Group': grp,
                'Intercept': intercept,
                'Intercept_SE': intercept_se,
                'Intercept_t': intercept_t,
                'Intercept_p': intercept_p,
                'Slope': slope,
                'Slope_SE': slope_se,
                'Slope_t': slope_t,
                'Slope_p': slope_p,
                'R2': r2,
                'R2_p': p_r2,
                'Block_Var': result.cov_re.iloc[0,0],
                'Block_p': p_block
            })

            preds = result.predict(pd.DataFrame({'Environmental_Index': env_vals}))
            ax.plot(env_vals, preds, label=f"{grp}", color=color_map[grp])

        stats_df = df.groupby(['Loc_Treatment', group_col])['Yield_kg_ha'].agg(
            mean='mean', std='std', count='count').reset_index()
        stats_df['se'] = stats_df['std'] / np.sqrt(stats_df['count'])
        env_index_map = df.groupby('Loc_Treatment')['Environmental_Index'].first().to_dict()

        for _, row in stats_df.iterrows():
            x_pos = env_index_map[row['Loc_Treatment']]
            ax.errorbar(x_pos, row['mean'], yerr=row['se'], fmt='o',
                        color=color_map[row[group_col]], capsize=5)

        # --- Environment-level ANOVA & Tukey ---
        for env in locs:
            sub_env = df[df['Loc_Treatment'] == env]
            groups_present = sub_env[group_col].dropna().unique()
            if len(groups_present) >= 2:
                groups_data = [g['Yield_kg_ha'].values for _, g in sub_env.groupby(group_col) if len(g) > 1]
                if len(groups_data) >= 2:
                    f_stat, p = f_oneway(*groups_data)
                    print(f"\nANOVA for {group_col} in {env}: F={f_stat:.3f}, p={p:.4f}")
                    means_table = sub_env.groupby(group_col)['Yield_kg_ha'].agg(['mean', 'std', 'count'])
                    means_table['se'] = means_table['std'] / np.sqrt(means_table['count'])
                    for grp, row in means_table.iterrows():
                        print(f"  {grp:<15} mean={row['mean']:.1f}, sd={row['std']:.1f}, n={int(row['count'])}")

                    if p < 0.05:
                        max_row = means_table.loc[(means_table['mean'] + means_table['se']).idxmax()]
                        x_pos = env_index_map[env]
                        y_star = max_row['mean'] + max_row['se'] * 0.4
                        ax.text(x_pos, y_star, '*', ha='center', va='bottom',
                                fontsize=18, color='red')

                    tukey = pairwise_tukeyhsd(endog=sub_env['Yield_kg_ha'],
                                              groups=sub_env[group_col],
                                              alpha=0.05)
                    print(tukey.summary())

        group_col_title = group_col.replace('_', ' ')
        ax.set_title(f'{group_col_title}', fontsize=20)
        if i == 0:
            ax.set_ylabel('Yield (kg/ha)', fontsize=18)
        else:
            ax.set_ylabel('')
        ax.grid(False)
        ax.legend(fontsize=16, title_fontsize=18, loc='upper left')
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

    fig.text(0.5, 0.04, 'Environmental Index (Mean Yield per Environment)', ha='center', fontsize=18)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig('yield_stability_plot.svg', format='svg', bbox_inches='tight')
    plt.show()

    fw_results_df = pd.DataFrame(fw_results)
    return fw_results_df

fw_results = fw_mixedlm_analysis_combined_plot(
    df_clean,
    ['Yield_group', 'Stability_group', 'Selection_group']
)

