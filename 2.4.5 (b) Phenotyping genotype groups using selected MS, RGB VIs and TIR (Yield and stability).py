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

df = pd.merge(df, yield_groups[['Code', 'Yield group', 'EMM', 'Stability group', 'Stability']], 
               on=['Code'], how='left')

os.chdir('..')
os.chdir('datasets\\meteo')

ggd = pd.read_csv("GDD_dates_all_locations.csv")

df = pd.merge(df,ggd[['Date','GDD']], on=['Date'], how='left')

ETo = pd.read_csv("ETo_prec_all_locations.csv")

df = pd.merge(df,ETo[['Date','Treatment','ETo (mm)' , 'water_input_cumulative']], 
              on=['Date', 'Treatment'], how='left')


# In[5]:


yield_indices_an = ['RDVI_UAV', 'NDVI_UAV','SAVI_UAV','GA_UAV', 'Hue_UAV', 'Hue_ground']
yield_indices_gf = ['v*_UAV','v*_ground', 'NDVI_ground',  'a*_ground',  'CSI_ground','Tveg_mean_UAV']
yield_indices_diff = ['NGRDIveg_ground','Lightness_UAV', 'u*_UAV', 'v*_UAV', 'GA_UAV', 'Tveg_mean_UAV']

stability_indices_an = ['RDVI_UAV','NDVI_ground', 'TCARI_UAV','NGRDIveg_UAV','CSI_UAV','b*_ground']
stability_indices_gf = ['v*_UAV', 'TCARI_UAV', 'NDVI_ground', 'Intensity_ground', 'u*_ground', 'GGA_ground']
stability_indices_diff = ['NGRDIveg_ground','RDVI_UAV','NDVI_ground', 'TGI_UAV',  'a*_ground', 'TGI_ground']


# In[7]:


def perform_anova_and_tukey(df, yield_indices, stability_indices):
    df_copy = df.copy()  
    results_valladolid_yield = {}
    results_aranjuez_yield = {}
    results_valladolid_stability = {}
    results_aranjuez_stability = {}

    locations = df_copy['Loc'].unique()

    for loc in locations:
        df_loc = df_copy[df_copy['Loc'] == loc].copy()  
        gdd_values = df_loc['GDD'].unique()

        results_dict_yield = results_valladolid_yield if loc == 'Valladolid' else results_aranjuez_yield
        results_dict_stability = results_valladolid_stability if loc == 'Valladolid' else results_aranjuez_stability

        for gdd in gdd_values:
            df_gdd = df_loc[df_loc['GDD'] == gdd].copy()

            for index in yield_indices:
                # Normalize the VI values for each index individually
                df_gdd.loc[:, index] = df_gdd.groupby(['Loc', 'Date', 'Treatment'])[index].transform(
                    lambda x: (x - x.mean()) / abs(x.mean()) * 100
                )

                groups = [df_gdd[df_gdd['Yield group'] == group][index].dropna() for group in df_gdd['Yield group'].unique()]
                groups = [g for g in groups if len(g) > 0]  # Keep only non-empty groups

                if len(groups) > 1:  
                    model = stats.f_oneway(*groups)

                    if model.pvalue < 0.05:
                        tukey = pairwise_tukeyhsd(df_gdd[index], df_gdd['Yield group'])
                        significant_groups = []

                        for res in tukey.summary().data[1:]:
                            g1, g2, pval = res[0], res[1], res[3]
                            if pval < 0.05:
                                significant_groups.append(f"{g1[0]}{g1[1]} - {g2[0]}{g2[1]}")

                        if len(significant_groups) == 3:
                            significant_groups = ["All"]

                        if significant_groups:
                            results_dict_yield.setdefault(index, []).append({
                                'GDD': gdd,
                                'p-value': round(model.pvalue, 6),
                                'significant_groups': ', '.join(significant_groups)
                            })
                    else:
                        results_dict_yield.setdefault(index, []).append({
                            'GDD': gdd,
                            'p-value': '',
                            'significant_groups': ''
                        })

            # For stability indices
            for index in stability_indices:
                # Normalize the VI values for each index individually
                df_gdd.loc[:, index] = df_gdd.groupby(['Loc', 'Date', 'Treatment'])[index].transform(
                    lambda x: (x - x.mean()) / abs(x.mean()) * 100
                )

                groups = [df_gdd[df_gdd['Stability group'] == group][index].dropna() for group in df_gdd['Stability group'].unique()]
                groups = [g for g in groups if len(g) > 0]

                if len(groups) > 1:
                    model = stats.f_oneway(*groups)

                    if model.pvalue < 0.05:
                        tukey = pairwise_tukeyhsd(df_gdd[index], df_gdd['Stability group'])
                        significant_groups = []

                        for res in tukey.summary().data[1:]:
                            g1, g2, pval = res[0], res[1], res[3]
                            if pval < 0.05:
                                significant_groups.append(f"{g1[0]}{g1[1]} - {g2[0]}{g2[1]}")

                        if len(significant_groups) == 3:
                            significant_groups = ["All"]

                        if significant_groups:
                            results_dict_stability.setdefault(index, []).append({
                                'GDD': gdd,
                                'p-value': round(model.pvalue, 6),
                                'significant_groups': ', '.join(significant_groups)
                            })
                    else:
                        results_dict_stability.setdefault(index, []).append({
                            'GDD': gdd,
                            'p-value': round(model.pvalue, 6),
                            'significant_groups': ''
                        })

    return results_valladolid_yield, results_aranjuez_yield, results_valladolid_stability, results_aranjuez_stability

valladolid_yield_diff, aranjuez_yield_diff, valladolid_stability_diff, aranjuez_stability_diff = perform_anova_and_tukey(
    df, yield_indices_diff, stability_indices_diff)
valladolid_yield_an, aranjuez_yield_an, valladolid_stability_an, aranjuez_stability_an = perform_anova_and_tukey(
    df, yield_indices_an, stability_indices_an)
valladolid_yield_gf, aranjuez_yield_gf, valladolid_stability_gf, aranjuez_stability_gf = perform_anova_and_tukey(
    df, yield_indices_gf, stability_indices_gf)


# In[8]:


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

substitute_gdd_values(aranjuez_yield_an, gdd_substitutions)
substitute_gdd_values(aranjuez_stability_an, gdd_substitutions)
# print(valladolid_yield_an['SAVI_UAV'])

substitute_gdd_values(aranjuez_yield_diff, gdd_substitutions)
substitute_gdd_values(aranjuez_stability_diff, gdd_substitutions)

substitute_gdd_values(aranjuez_yield_gf, gdd_substitutions)
substitute_gdd_values(aranjuez_stability_gf, gdd_substitutions)


# In[13]:


os.chdir('..\\..\\results\\figures\\phenotyping_yield_and_stability')


# In[15]:


def plot_ideotype(df, yield_indices, stability_indices, dataset_name, treatment=None, 
                  valladolid_yield_results=None, aranjuez_yield_results=None,
                  valladolid_stability_results=None, aranjuez_stability_results=None):
    if treatment:
        df = df[df['Treatment'] == treatment].copy()

    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(14, 20), sharex=True, constrained_layout=True)

    yield_hue_order = ['Low yield', 'Intermediate yield', 'High yield']
    stability_hue_order = ['Low stability', 'Intermediate stability', 'High stability']

    yield_palette = sns.color_palette("colorblind", len(yield_hue_order))
    stability_palette = sns.color_palette("colorblind", len(stability_hue_order))

    yield_handles, yield_labels = None, None
    stability_handles, stability_labels = None, None

    for i, (yield_index, stability_index) in enumerate(zip(yield_indices, stability_indices)):
        for col, (index, group_col, palette, hue_order, results_dict, label_color, xy_offset) in enumerate([
            (yield_index, 'Yield group', yield_palette, yield_hue_order, 
             valladolid_yield_results, '#D32F2F', (8, 18)),  
            (stability_index, 'Stability group', stability_palette, stability_hue_order, 
             valladolid_stability_results, '#9B59B6', (8, 28))  
        ]):
            ax = axes[i, col]
            plot_data = df.copy()  

            plot_data[index] = plot_data.groupby(['Treatment', 'Date'])[index].transform(
                lambda x: (x - x.mean()) / abs(x.mean()) * 100
            )

            plot_data_grouped = (
                plot_data.groupby([group_col, 'GDD', 'Loc'])[index]
                .agg(mean='mean', se=lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
                .reset_index()
            )

            sns.lineplot(
                data=plot_data_grouped, x='GDD', y='mean', hue=group_col,
                marker='o', palette=palette, hue_order=hue_order, err_style=None, ax=ax
            )

            for j, group in enumerate(hue_order):
                group_data = plot_data_grouped[plot_data_grouped[group_col] == group]
                ax.errorbar(
                    group_data['GDD'], group_data['mean'], yerr=group_data['se'], fmt='none',
                    ecolor=palette[j], alpha=0.7, capsize=3
                )

            ax.set_title(index, fontsize=14)
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.legend().remove()

            for location, results_dict, label_color, xy_offset in [
                ('Valladolid', valladolid_yield_results if col == 0 else valladolid_stability_results, '#D32F2F', (-20, 10)),  
                ('Aranjuez', aranjuez_yield_results if col == 0 else aranjuez_stability_results, '#9B59B6', (-20, 35))  
            ]:
                if results_dict and index in results_dict:
                    for result_dict in results_dict[index]:  
                        gdd_value = result_dict['GDD']
                        significant_groups = result_dict['significant_groups']

                        gdd_data = plot_data_grouped[np.isclose(plot_data_grouped['GDD'], gdd_value, atol=1e-3)]
                        if not gdd_data.empty:
                            x_pos = gdd_data['GDD'].values[0]
                            y_pos = gdd_data['mean'].mean()  

                            ax.annotate(
                                "\n".join([group.strip() for group in significant_groups.split(",")]), 
                                (x_pos, y_pos), 
                                textcoords="offset points", 
                                xytext=xy_offset,  
                                fontsize=12, color=label_color
                            )

            if col == 0 and yield_handles is None:
                yield_handles, yield_labels = ax.get_legend_handles_labels()
            elif col == 1 and stability_handles is None:
                stability_handles, stability_labels = ax.get_legend_handles_labels()

    if yield_handles and stability_handles:
        axes[0, 0].legend(yield_handles, yield_labels, title='Yield group', loc='upper center',
                          bbox_to_anchor=(0.5, 1.35), frameon=False, ncol=3, fontsize=12, title_fontsize=12)
        axes[0, 1].legend(stability_handles, stability_labels, title='Stability group', loc='upper center', 
                          bbox_to_anchor=(0.5, 1.35), frameon=False, ncol=3, fontsize=12, title_fontsize=12)

    fig.text(0.5, -0.01, 'Growing Degree Days', ha='center', fontsize=16)
    fig.text(-0.01, 0.5, 'Normalized percentage difference to the mean', va='center', rotation='vertical', fontsize=16)

    fig.suptitle(f"Seasonal reflectance phenotypes for yield and yield stability \n (features selected at {dataset_name})", fontsize=18, y=1.03)

    fig.savefig(f"{dataset_name}_yield_stability_phenotypes.png", dpi=100, bbox_inches='tight')
    plt.show()

dataset = df[df['Loc'] == 'Valladolid']
plot_ideotype(dataset, yield_indices_an, stability_indices_an, 'anthesis', 
              valladolid_yield_results=valladolid_yield_an, aranjuez_yield_results=aranjuez_yield_an,
              valladolid_stability_results=valladolid_stability_an, aranjuez_stability_results=aranjuez_stability_an)

plot_ideotype(dataset, yield_indices_diff, stability_indices_diff, 'index-difference', 
              valladolid_yield_results=valladolid_yield_diff, aranjuez_yield_results=aranjuez_yield_diff,
              valladolid_stability_results=valladolid_stability_diff, aranjuez_stability_results=aranjuez_stability_diff)
plot_ideotype(dataset, yield_indices_gf, stability_indices_gf, 'grain filling', 
              valladolid_yield_results=valladolid_yield_gf, aranjuez_yield_results=aranjuez_yield_gf,
              valladolid_stability_results=valladolid_stability_gf, aranjuez_stability_results=aranjuez_stability_gf)

