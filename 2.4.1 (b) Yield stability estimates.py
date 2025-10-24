#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.formula.api import mixedlm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression


# In[2]:


# Just run once
notebook_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(notebook_dir, os.pardir))
os.chdir(parent_dir)
print("Working directory set to:", os.getcwd())


# In[3]:


os.chdir('datasets')
print(os.getcwd())

df = pd.read_csv('holistic_wheat_merged_multi.csv')

df = df[(df['Date'] == '2024-05-14') | (df['Date'] == '2024-05-29')]

df_y = pd.read_csv('holistic_wheat_yield_merged.csv')

df['PN'] = df['PN'].astype(int)
df_y['PN'] = df_y['PN'].astype(int)

df = pd.merge(df, df_y[['Loc', 'PN', 'Treatment', 'Code', 'Yield (kg/ha)']], 
              on=['Loc', 'PN', 'Treatment', 'Code'], how='left')

# Combine 'Loc' and 'Treatment' into a single interaction factor: environment
df['Loc_Treatment'] = df['Loc'] + "_" + df['Treatment']

df = df.dropna(subset=['Yield (kg/ha)'])

# Yield distribution plots #
os.chdir('..')
os.chdir('results')

df['Code'] = df['Code'].astype('category')
df['Loc_Treatment'] = df['Loc_Treatment'].astype('category')

df = df.rename(columns={'Yield (kg/ha)': 'Yield_kg_ha'})

unique_loc_treatments = df['Loc_Treatment'].unique()


# In[4]:


#### STABILITY CALCULATION  #####

# Calculate Environmental Index (Mean Yield for Each Location-Treatment)
df['Environmental_Index'] = df.groupby('Loc_Treatment', observed=False)['Yield_kg_ha'].transform('mean')
df['Loc_Treatment_Block'] = df['Loc_Treatment'].astype(str) + "_B" + df['B'].astype(str)


# Fit Linear Mixed Model including Block as random effect
# Yi = β0 + β1 * EI + β2 * G + β3 * (EI * G) + b_block + εi
formula = "Yield_kg_ha ~ Environmental_Index * Code"
model = smf.mixedlm(formula=formula, data=df, groups=df["Loc_Treatment_Block"])  # 'Loc_Treatment_Block' is unique block ID
results = model.fit(reml=False)

# Extract Coefficients for Stability Analysis
coeffs = results.fe_params  # fixed effects only

# Separate Main Effects and Interaction Terms
main_effect_env = coeffs["Environmental_Index"]  # β1
interaction_terms = coeffs.filter(like="Environmental_Index:Code")  # β3 terms
genotype_effects = coeffs.filter(like="Code[T.")  # β2 terms
genotype_effects = genotype_effects[~genotype_effects.index.str.contains('Environmental_Index')].reset_index()

# Create a DataFrame for Stability Coefficients
stability_df = interaction_terms.reset_index()
stability_df.columns = ['Term', 'Stability']
stability_df['Code'] = stability_df['Term'].str.extract(r'Code\[(.+?)\]')

# Add Reference Genotype (Aceres) Stability
acer_slope = main_effect_env  

# Adjust for Correct Slope Interpretation
stability_df['Stability'] += main_effect_env  # Add main effect to all slopes

# Reference genotype slope is the main effect of Environmental_Index
stability_df = pd.concat([
    stability_df,
    pd.DataFrame({'Term': ['Environmental_Index:Code[T.Aceres]'], 'Stability': [acer_slope], 'Code': ['Aceres']})
], ignore_index=True)

# Remove 'T.' from Code values
stability_df['Code'] = stability_df['Code'].str.replace(r'^T\.', '', regex=True)

# Extract Intercepts for Each Genotype
genotype_effects.columns = ['Term', 'Beta2']
genotype_effects['Code'] = genotype_effects['Term'].str.extract(r'Code\[(.+?)\]')

# Calculate Intercepts: β0 + β2 for each genotype
reference_intercept = coeffs["Intercept"]  # Baseline intercept (β0)
genotype_effects['Intercept'] = reference_intercept + genotype_effects['Beta2']

# Add Reference Genotype Intercept (Aceres)
reference_row = pd.DataFrame({'Code': ['Aceres'], 'Intercept': [reference_intercept]})
intercepts = pd.concat([genotype_effects[['Code', 'Intercept']], reference_row], ignore_index=True)

# Remove 'T.' from Code values
intercepts['Code'] = intercepts['Code'].str.replace(r'^T\.', '', regex=True)

# Combine Stability and Intercepts into Final Table
stability_df = pd.merge(stability_df[['Code', 'Stability']], intercepts[['Code', 'Intercept']], on='Code')
stability_df = stability_df.sort_values(by='Stability').reset_index(drop=True)

print(stability_df)


# In[5]:


sorted_df = stability_df.sort_values(by='Stability')

# Determine middle two indices
mid_start = (len(sorted_df) // 2) - 1  # Start index for middle two
mid_end = mid_start + 2  # Select two middle values

# List of top three, middle two, and bottom three stable genotypes
top_bot_genotypes = (
    sorted_df['Code'].head(3).tolist() +  # Top 3
    sorted_df['Code'].iloc[mid_start:mid_end].tolist() +  # Middle 2
    sorted_df['Code'].tail(3).tolist()  # Bottom 3
)

print("\nTop, Middle, and Bottom Genotypes List:", top_bot_genotypes)


# In[6]:


genotypes_toplot =['LGDE15-1151-B ( LG Quovadis)', 'Don Valentin', 'Odisseo', 'Semidou', 'Makrodur', 'Duramonte']
def plot_genotype_stability(df, top_bot_genotypes, s_df):
    """
    Plots yield vs environmental index for genotypes selected based on stability (Max & Min Stability).
    
    Parameters:
    - df (DataFrame): Data with 'Yield_kg_ha', 'Environmental_Index', and 'Code'.
    - top_bot_genotypes (list): Genotypes with highest and lowest stability.
    - s_df (DataFrame): Pre-calculated stability and intercept values for each genotype.
    """
    fig, ax = plt.subplots(figsize=(8, 6))  
    
    num_colors = len(top_bot_genotypes)
    palette = sns.color_palette("colorblind", n_colors=num_colors)

    df_filtered = df[df['Code'].isin(top_bot_genotypes)].copy()
    df_filtered['Code'] = df_filtered['Code'].astype(str).str.strip()

    sns.scatterplot(data=df_filtered, x='Environmental_Index', y='Yield_kg_ha', hue='Code', palette=palette, s=50, ax=ax)

    unique_codes = df_filtered['Code'].unique()
    color_map = dict(zip(unique_codes, palette))

    handles, labels = [], []
    
    for i, genotype in enumerate(top_bot_genotypes):
        genotype = genotype.strip() 
        if genotype not in color_map:
            print(f"Warning: Genotype '{genotype}' not found in color map. Skipping.")
            continue
        
        try:
            genotype_slope = s_df.loc[s_df['Code'].str.strip() == genotype, 'Stability'].values[0]
            genotype_intercept = s_df.loc[s_df['Code'].str.strip() == genotype, 'Intercept'].values[0]
        except IndexError:
            print(f"Warning: Missing stability/intercept data for '{genotype}'. Skipping.")
            continue

        x_vals = np.linspace(df_filtered['Environmental_Index'].min(), df_filtered['Environmental_Index'].max(), 100)
        y_vals = genotype_intercept + genotype_slope * x_vals

        genotype_name_for_legend = genotype.split('(')[-1].split(')')[0].strip()

        if i == 0:
            line_label = f"{genotype_name_for_legend} (Max Stability, Slope: {genotype_slope:.2f}, Intercept: {genotype_intercept:.2f})"
            line_style = '-'  
        elif i == len(top_bot_genotypes) - 1:
            line_label = f"{genotype_name_for_legend} (Min Stability, Slope: {genotype_slope:.2f}, Intercept: {genotype_intercept:.2f})"
            line_style = '-'  
        else:
            line_label = f"{genotype_name_for_legend} (Slope: {genotype_slope:.2f}, Intercept: {genotype_intercept:.2f})"
            line_style = '--'  

        line, = ax.plot(x_vals, y_vals, line_style, label=line_label, lw=2, color=color_map.get(genotype, 'gray'))

        handles.append(line)
        labels.append(line_label)

    ax.set_title("Yield vs Environmental Index")
    ax.set_xlabel('Environmental Index (kg/ha) (Mean Yield for Location-Treatment)')
    ax.set_ylabel('Yield (kg/ha)')
    ax.legend(handles=handles, labels=labels, loc='best', fontsize=9)
    ax.grid(True)

    plt.tight_layout()
    #plt.savefig(f"stability_plot.png", dpi=300)
    plt.show()

os.chdir('figures')
plot_genotype_stability(df, genotypes_toplot, stability_df)


# In[7]:


# Classify Genotypes into Stability Groups
# Calculate Quartiles
upper_quartile = stability_df["Stability"].quantile(0.75)
lower_quartile = stability_df["Stability"].quantile(0.25)

# Classify stability according to quartiles. Genotypes with lower than average slopes are the most stable
def classify_stability(value):
    if value >= upper_quartile:
        return "Low stability"
    elif value <= lower_quartile:
        return "High stability"
    else:
        return "Intermediate stability"

stability_df["Stability group"] = stability_df["Stability"].apply(classify_stability)

os.chdir('..')
stability_df.to_csv("yield_stability_groups.csv", index=False)

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=stability_df,
    x="Stability group",
    y="Stability",
    hue="Stability group", 
    order=["Low stability", "Intermediate stability", "High stability"],
    palette="pastel",
    dodge=False 
)
plt.title("Yield Stability Classification")
plt.ylabel("Stability Coefficient (β3)")
plt.xlabel("Stability Group")
plt.legend([], [], frameon=False) 
plt.tight_layout()

#plt.savefig(f"{output_dir}/yield_stability_boxplot.png", dpi=300)

plt.show()
print(stability_df)

# --- Print group means and SDs ---
summary_stats = (
    stability_df.groupby("Stability group")["Stability"]
    .agg(["mean", "std", "count"])
    .reindex(["Low stability", "Intermediate stability", "High stability"])
)

print("\n Stability group summary statistics:")
print(summary_stats.round(3))

