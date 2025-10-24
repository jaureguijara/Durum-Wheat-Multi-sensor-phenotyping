#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ================================================================
# CORRELATION HEATMAPS: VI vs Yield and VI Stability vs Yield Stability
# ================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === 1. Navigate to results directory ===
notebook_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(notebook_dir, os.pardir))
os.chdir(os.path.join(parent_dir, "results"))

# === 2. Read the correlation CSVs ===
yield_corr = pd.read_csv("VI_yield_correlations_all.csv")
stab_corr  = pd.read_csv("VI_stability_stability_correlations.csv")

# === 3. Clean dataset names ===
def clean_dataset_name(name):
    name = str(name).replace("_", " ").strip()
    if "Index" in name and "Difference" in name:
        return "Index-difference"
    elif "Grain" in name:
        return "Grain Filling"
    elif "Anthesis" in name:
        return "Anthesis"
    return name

yield_corr["Dataset"] = yield_corr["Dataset"].apply(clean_dataset_name)
stab_corr["Dataset"]  = stab_corr["Dataset"].apply(clean_dataset_name)


def plot_correlation_heatmaps(df, value_col, title, filename):

    anthesis_df = df[df["Dataset"].str.contains("Anthesis", case=False, na=False)]
    anthesis_df = anthesis_df.sort_values(value_col, ascending=False)
    index_order = anthesis_df["Index"].tolist()


    all_indices = set(df["Index"].unique())
    missing_indices = sorted(all_indices - set(index_order))
    index_order.extend(missing_indices)


    datasets = ["Anthesis", "Grain Filling", "Index-difference"]
    datasets = [d for d in datasets if d in df["Dataset"].unique()]  #
    num_datasets = len(datasets)

    fig, axes = plt.subplots(
        1, num_datasets, figsize=(5 * num_datasets, 14), sharey=True
    )
    cmap = sns.color_palette("vlag", as_cmap=True)

    if num_datasets == 1:
        axes = [axes]

    for i, dataset in enumerate(datasets):
        subset = df[df["Dataset"] == dataset].copy()
        subset = subset.set_index("Index").reindex(index_order)
        data = subset[[value_col]]

        heat = sns.heatmap(
            data,
            ax=axes[i],
            cmap=cmap,
            center=0,
            annot=True,
            fmt=".2f",
            cbar=(i == num_datasets - 1),
            annot_kws={"size": 14, "weight": "bold"},
            linewidths=0.5,
            linecolor="gray"
        )

        # --- Adjust colorbar ---
        if i == num_datasets - 1:
            cbar = heat.collections[0].colorbar
            cbar.ax.tick_params(labelsize=14)

        axes[i].set_title(dataset, fontsize=16, pad=20, weight="bold")
        axes[i].set_xlabel("")
        if i == 0:
            axes[i].set_ylabel("", fontsize=14, labelpad=10)
        else:
            axes[i].set_ylabel("")

        axes[i].tick_params(axis='x', bottom=False, labelbottom=False)
        axes[i].tick_params(axis='y', labelsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle(title, fontsize=22, y=1, weight="bold")
    plt.savefig(filename, dpi=350, bbox_inches="tight")
    plt.show()


# ================================================================
# 5. Plot 1: Correlation with Yield
# ================================================================
os.chdir('figures')

plot_correlation_heatmaps(
    df=yield_corr,
    value_col="Correlation_with_Yield",
    title="RGB & MS VI and TIR correlation with yield",
    filename="VI_TIR_yield_correlations.png"
)

# ================================================================
# 6. Plot 2: Stability Correlation with Yield Stability
# ================================================================

plot_correlation_heatmaps(
    df=stab_corr,
    value_col="Correlation_with_Stability",
    title="RGB & MS VI and TIR stability correlation with yield stability",
    filename="VI_TIR_stability_correlations.png"
)

