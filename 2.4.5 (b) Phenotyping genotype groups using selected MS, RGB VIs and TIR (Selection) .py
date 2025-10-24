#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings

np.seterr(all='ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered in reduce.*')

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


stability_df = pd.read_csv('yield_stability_groups.csv')
yield_groups = pd.read_csv('yield_groups_based_on_emms.csv')

yield_groups = pd.merge(yield_groups, stability_df[['Code', 'Stability', 'Stability group']],
                        on=['Code'], how ='left')

df = pd.merge(df, yield_groups[['Code', 'Yield_group', 'emmean', 'Stability group', 'Stability']], 
               on=['Code'], how='left')

os.chdir('..')
os.chdir('datasets\\meteo')

ggd = pd.read_csv("GDD_dates_all_locations.csv")

df = pd.merge(df,ggd[['Date','GDD']], on=['Date'], how='left')

print(os.getcwd())
os.chdir('..')
os.chdir('..\\results\\feature_selection')

yield_pred_df = pd.read_csv("yield_predictions_an.csv")

stability_pred_df = pd.read_csv("stability_loo_an.csv")

df = pd.merge(df, yield_pred_df[['Code', 'Yield group predicted']],on=['Code'], how='left')
df = pd.merge(df, stability_pred_df[['Code', 'Stability group predicted']],on=['Code'], how='left')

df['Selection observed'] = np.where(
    (df['Yield_group'] == 'Low yield') | (df['Stability group'] == 'Low stability'),
    'Discard',
    'Keep'
)

df['Selection predicted'] = np.where(
    (df['Yield group predicted'] == 'Low yield') | (df['Stability group predicted'] == 'Low stability'),
    'Discard',
    'Keep'
)


# In[3]:


df.head()


# In[4]:


yield_indices_an = ['RDVI_UAV', 'NDVI_UAV','SAVI_UAV','GA_UAV', 'Hue_UAV', 'Hue_ground']
yield_indices_gf = ['v*_UAV','v*_ground', 'NDVI_ground',  'a*_ground',  'CSI_ground','Tveg_mean_UAV']
yield_indices_diff = ['NGRDIveg_ground','Lightness_UAV', 'u*_UAV', 'v*_UAV', 'GA_UAV', 'Tveg_mean_UAV']

stability_indices_an = ['GGA_UAV', 'NGRDI_UAV', 'NDVI_UAV', 'SAVI_UAV', 'Tveg_mean_UAV', 'u*_ground']
stability_indices_gf = ['Hue_UAV', 'Saturation_UAV', 'Tveg_mean_UAV', 'NDVI_ground', 'v*_ground']
stability_indices_diff = ['Saturation_UAV', 'NGRDIveg_UAV', 'NDVI_UAV', 'Lightness_ground', 'a*_ground', 'TGI_ground']


# In[5]:


# Combine and deduplicate per stage
indices_an = list(set(yield_indices_an + stability_indices_an))
indices_gf = list(set(yield_indices_gf + stability_indices_gf))
indices_diff = list(set(yield_indices_diff + stability_indices_diff))

# Combine all indices and deduplicate
indices_all = list(set(indices_an + indices_gf + indices_diff))

print(indices_an, len(indices_an))
print(indices_gf, len(indices_gf))
print(indices_diff, len(indices_diff))
print(indices_all, len(indices_all))


# In[6]:


# List reordered 
indices_all = ['RDVI_UAV', 'SAVI_UAV', 'NDVI_UAV', 'NDVI_ground', 
               'Hue_UAV', 'Lightness_UAV', 'Saturation_UAV', 'a*_ground',
               'Hue_ground', 'Lightness_ground', 'v*_UAV', 'u*_UAV',
               'GA_UAV', 'NGRDI_UAV',  'v*_ground', 'u*_ground', 
               'GGA_UAV', 'NGRDIveg_ground', 'TGI_ground', 
               'CSI_ground','NGRDIveg_UAV',   'Tveg_mean_UAV']


# In[7]:


def perform_anova_and_tukey_selection(df, index_list):
    df_copy = df.copy()
    results_valladolid = {'significant_groups': []}
    results_aranjuez = {'significant_groups': []}

    rows_valladolid = []
    rows_aranjuez = []

    for loc in df_copy['Loc'].unique():
        df_loc = df_copy[df_copy['Loc'] == loc].copy()
        gdd_values = df_loc['GDD'].unique()
        result_list = rows_valladolid if loc == 'Valladolid' else rows_aranjuez
        results_dict = results_valladolid if loc == 'Valladolid' else results_aranjuez

        for gdd in gdd_values:
            df_gdd = df_loc[df_loc['GDD'] == gdd].copy()

            for index in index_list:
                try:
                    # Normalize index within Date and Treatment
                    df_gdd[index] = df_gdd.groupby(['Loc', 'Date', 'Treatment'])[index].transform(
                        lambda x: (x - x.mean()) / abs(x.mean()) * 100
                    )

                    # Get means by group
                    means = df_gdd.groupby('Selection observed')[index].mean()
                    if 'Keep' in means and 'Discard' in means:
                        keep_mean = means['Keep']
                        discard_mean = means['Discard']
                        perc_diff = abs(keep_mean - discard_mean)
                        
                    else:
                        keep_mean, discard_mean, perc_diff = None, None, None

                    # Get groups for ANOVA
                    groups = [
                        df_gdd[df_gdd['Selection observed'] == group][index].dropna()
                        for group in df_gdd['Selection observed'].dropna().unique()
                    ]
                    groups = [g for g in groups if len(g) > 0]

                    if len(groups) > 1:
                        model = stats.f_oneway(*groups)
                        pval = round(model.pvalue, 6)
                        sig = '*' if pval < 0.05 else ''
                        
                        if sig:
                            tukey = pairwise_tukeyhsd(df_gdd[index], df_gdd['Selection observed'])

                        results_dict['significant_groups'].append({
                            'GDD': gdd,
                            'Index': index,
                            'p-value': pval,
                            'significant_groups': sig
                        })

                        result_list.append({
                            'Location': loc,
                            'GDD': gdd,
                            'Index': index,
                            'Keep Mean': round(keep_mean, 2) if keep_mean is not None else None,
                            'Discard Mean': round(discard_mean, 2) if discard_mean is not None else None,
                            'Difference % (relative difference)': round(perc_diff, 2) if perc_diff is not None else None,
                            'p-value': pval,
                            'Significant': sig
                        })

                except Exception as e:
                    print(f"Skipping index {index} at GDD {gdd} in {loc} due to error: {e}")
                    continue

    table_valladolid = pd.DataFrame(rows_valladolid)
    table_aranjuez = pd.DataFrame(rows_aranjuez)

    return results_valladolid, results_aranjuez, table_valladolid, table_aranjuez

valladolid_all, aranjuez_all, table_valladolid_all, table_aranjuez_all = perform_anova_and_tukey_selection(df, indices_all)


# In[8]:


merged_table_all = pd.concat([table_valladolid_all, table_aranjuez_all], ignore_index=True)
merged_table_all = merged_table_all.sort_values(by=['Index', 'GDD'])
print(merged_table_all)
merged_table_all.to_csv('merged_selection_differences_normalized.csv', index=False)


# In[9]:


os.getcwd()


# In[10]:


def substitute_gdd_values(aranjuez_dict, gdd_substitutions):
    """
    ONLY FOR PLOTTING PURPOSES. 
    This function takes an Aranjuez dictionary and a GDD substitution mapping,
    and substitutes the GDD values in the dictionary based on the provided mappings.
    
    :param aranjuez_dict: Dictionary containing the GDD values to be substituted
    :param gdd_substitutions: Dictionary with old GDD values as keys and new GDD values as values
    :return: The updated dictionary with substituted GDD values
    """
    for index, entries in aranjuez_dict.items():
        for entry in entries:
            if entry['GDD'] in gdd_substitutions:
                entry['GDD'] = gdd_substitutions[entry['GDD']]
    

# GDD substitution mapping
gdd_substitutions = {1496.85: 1489.18, 1819.82: 1735.46}

# Substitute the GDD values in the Aranjuez dictionary
substitute_gdd_values(aranjuez_all, gdd_substitutions)


# In[11]:


print(aranjuez_all)


# In[12]:


os.getcwd()


# In[13]:


#os.chdir("..")
os.chdir("..")
os.chdir('figures\\phenotyping_selection')


# In[14]:


dataset = df[df['Loc'] == 'Valladolid'].copy()

def plot_selection_groups_all(df, indices, valladolid_results, aranjuez_results):
    ncols = 4
    nrows = 6 
    total_axes = ncols * nrows

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 24), constrained_layout=True)
    axes = axes.flatten()

    selection_hue_order = ['Keep', 'Discard']
    selection_palette = sns.color_palette("colorblind", len(selection_hue_order))

    index_sources = {
        'A': yield_indices_an + stability_indices_an,
        'GF': yield_indices_gf + stability_indices_gf,
        'ID': yield_indices_diff + stability_indices_diff
    }

    # Reserve (blank) the 4th column in rows 5 and 6 (0-indexed rows 4 and 5 -> axes 19 and 23)
    reserved_axes = {4 * ncols + (ncols - 1), 5 * ncols + (ncols - 1)}  # {19, 23}
    # Create a list of axes indices to fill with plots (skip reserved)
    axes_to_fill = [ax_idx for ax_idx in range(total_axes) if ax_idx not in reserved_axes]

    # We'll map each index -> next available axes_to_fill
    if len(indices) > len(axes_to_fill):
        print("Warning: more indices than available subplots (after reserving blanks). Some indices will be skipped.")

    for idx_i, index in enumerate(indices):
        if idx_i >= len(axes_to_fill):
            break  # no more available subplot axes
        ax_idx = axes_to_fill[idx_i]
        ax = axes[ax_idx]

        plot_data = df.copy()
        try:
            plot_data.loc[:, index] = plot_data.groupby(['Date', 'Treatment'])[index].transform(
                lambda x: (x - x.mean()) / abs(x.mean()) * 100
            )
        except Exception as e:
            print(f"Skipping index {index} due to error: {e}")
            ax.axis('off')
            continue

        plot_data_grouped = (
            plot_data.groupby(['Selection observed', 'GDD'])[index]
            .agg(mean='mean', se=lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
            .reset_index()
        )

        sns.lineplot(
            data=plot_data_grouped, x='GDD', y='mean', hue='Selection observed',
            marker='o', palette=selection_palette, hue_order=selection_hue_order, ax=ax
        )

        for j, group in enumerate(selection_hue_order):
            group_data = plot_data_grouped[plot_data_grouped['Selection observed'] == group]
            ax.errorbar(
                group_data['GDD'], group_data['mean'], yerr=group_data['se'], fmt='none',
                ecolor=selection_palette[j], alpha=0.7, capsize=3
            )

        # Significance stars 
        star_colors = ['#d62728', '#9467bd']  
        for loc_results, color, x_offset in [
            (valladolid_results['significant_groups'], star_colors[0], -8),
            (aranjuez_results['significant_groups'], star_colors[1], 12)
        ]:
            for result in loc_results:
                if result['Index'] == index and result['significant_groups'] == '*':
                    gdd = result['GDD']
                    gdd_data = plot_data_grouped.loc[np.isclose(plot_data_grouped['GDD'], gdd, atol=1e-3)].copy()
                    if gdd_data.empty:
                        continue

                    gdd_data.loc[:, 'y_pos'] = gdd_data['mean'] + gdd_data['se']
                    max_row = gdd_data.loc[gdd_data['y_pos'].idxmax()]

                    y_star = max_row['y_pos'] * 0.97
                    ax.text(
                        gdd + x_offset, y_star, '*', color=color,
                        ha='center', va='bottom', fontsize=14, fontweight='bold'
                    )

        ax.set_title(index, fontsize=22)

        origin_labels = [label for label, idx_set in index_sources.items() if index in idx_set]
        if origin_labels:
            ax.text(
                0.1, 0.975, ', '.join(origin_labels),
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=16, fontstyle='italic', color='dimgray'
            )

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.legend().remove()

        x_tick_axes = {15, 20, 21, 22}
        if ax_idx in x_tick_axes:
            ax.tick_params(axis='x', labelbottom=True)
        else:
            ax.tick_params(axis='x', labelbottom=False)


    # Turn off reserved axes (blank them) and use one (axes[23]) for legend
    for ra in reserved_axes:
        axes[ra].axis('off')

    # Place legend in the bottom-right reserved subplot (axes[23])
    legend_ax = axes[19]
    legend_ax.axis('off')  # keep it visually blank behind the legend
    handles = [plt.Line2D([0], [0], color=selection_palette[i], marker='o', 
                         linestyle='-', linewidth=2, markersize=8) 
              for i in range(len(selection_hue_order))]

    legend_ax.legend(handles, selection_hue_order, 
                    title='Selection', 
                    loc='center',
                    frameon=True,
                    fontsize=22, 
                    title_fontsize=20, 
                    borderpad=1.5, 
                    edgecolor='black')

    fig.text(0.5, -0.02, 'Growing Degree Days', ha='center', fontsize=30)
    fig.text(-0.015, 0.5, 'Normalized percentage difference to the mean', va='center', rotation='vertical', fontsize=30)

    fig.suptitle("Seasonal reflectance phenotypes for genotype selection", fontsize=32, y=1.03)

    #fig.savefig("phenotypes_all_selection_groups_normalized.png", dpi=100, bbox_inches='tight')
    plt.savefig('phenotypes_all_selection_groups_normalized.svg', format='svg', bbox_inches='tight')

    plt.show()


plot_selection_groups_all(dataset, indices_all, valladolid_all, aranjuez_all)

