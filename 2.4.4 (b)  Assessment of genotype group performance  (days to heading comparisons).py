#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# In[2]:


notebook_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(notebook_dir, os.pardir))
os.chdir(parent_dir)
print("Working directory set to:", os.getcwd())
os.chdir('results')


# In[3]:


selection_groups=pd.read_csv("selection_groups.csv")


# In[4]:


os.chdir('..')
os.chdir('datasets')


# In[5]:


heading_df = pd.read_csv("heading_all.csv")


# In[6]:


df_h = pd.merge(
    heading_df,
    selection_groups[['Code', 'Yield group', 'Stability group', 'Selection']],
    on='Code',
    how='left'
)


df_h = df_h.dropna(subset=['esp', 'Selection'])

df_h['Selection'] = df_h['Selection'].astype(str)
df_h['Yield group'] = df_h['Yield group'].astype(str)
df_h['Stability group'] = df_h['Stability group'].astype(str)

df_h = df_h.rename(columns={
    'Yield group': 'Yield_group',
    'Stability group': 'Stability_group'
})

# ANOVA Yield group
model = smf.ols('esp ~ C(Yield_group)', data=df_h).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("### ANOVA ###")
print(anova_table)

print("\n### OLS model Yield group ###")
print(model.summary())

# Tukey HSD
tukey = pairwise_tukeyhsd(endog=df_h['esp'], groups=df_h['Yield_group'], alpha=0.05)
print("\n### Tukey HSD ###")
print(tukey)

tukey.plot_simultaneous()
plt.title("Tukey HSD: esp Yield group")
plt.xlabel("esp mean difference")
plt.show()

# ANOVA Stability group
model = smf.ols('esp ~ C(Stability_group)', data=df_h).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("### ANOVA ###")
print(anova_table)

print("\n### OLS model stability group ###")
print(model.summary())

# Tukey HSD
tukey = pairwise_tukeyhsd(endog=df_h['esp'], groups=df_h['Stability_group'], alpha=0.05)
print("\n### Tukey HSD ###")
print(tukey)

tukey.plot_simultaneous()
plt.title("Tukey HSD: esp Stability group")
plt.xlabel("esp mean difference")
plt.show()

# ANOVA Selection group
model = smf.ols('esp ~ C(Selection)', data=df_h).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("### ANOVA ###")
print(anova_table)

print("\n### OLS model selection group ###")
print(model.summary())

# Tukey HSD
tukey = pairwise_tukeyhsd(endog=df_h['esp'], groups=df_h['Selection'], alpha=0.05)
print("\n### Tukey HSD ###")
print(tukey)

tukey.plot_simultaneous()
plt.title("Tukey HSD: esp per Selection group")
plt.xlabel("esp mean difference")
plt.show()


# In[7]:


# Summary table: mean and SD for esp by each group type
summary_yield = df_h.groupby('Yield_group')['esp'].agg(['mean', 'std']).reset_index()
summary_yield['Group_type'] = 'Yield_group'

summary_stability = df_h.groupby('Stability_group')['esp'].agg(['mean', 'std']).reset_index()
summary_stability['Group_type'] = 'Stability_group'

summary_selection = df_h.groupby('Selection')['esp'].agg(['mean', 'std']).reset_index()
summary_selection['Group_type'] = 'Selection'

summary_yield = summary_yield.rename(columns={'Yield_group': 'Group'})
summary_stability = summary_stability.rename(columns={'Stability_group': 'Group'})
summary_selection = summary_selection.rename(columns={'Selection': 'Group'})

summary_table = pd.concat([summary_yield, summary_stability, summary_selection], ignore_index=True)

summary_table['mean'] = summary_table['mean'].round(2)
summary_table['std'] = summary_table['std'].round(2)

print("\n### Mean and SD for days to heading (esp) ###")
print(summary_table)

