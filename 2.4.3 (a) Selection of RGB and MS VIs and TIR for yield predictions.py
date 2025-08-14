#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
)

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

df = pd.merge(df, stability_df[['Code', 'Stability']],
                        on=['Code'], how ='left')

os.chdir('..')
os.chdir('datasets\\meteo')

ggd = pd.read_csv("GDD_dates_all_locations.csv")

df = pd.merge(df,ggd[['Date','GDD']], on=['Date'], how='left')

ETo = pd.read_csv("ETo_prec_all_locations.csv")

df = pd.merge(df,ETo[['Date','Treatment','ETo (mm)' , 'water_input_cumulative']], on=['Date', 'Treatment'], how='left')

# Normalize temperature variable 
df.loc[:, 'Tveg_mean_UAV'] = (
    (df['Tveg_mean_UAV'] - df.groupby(['Date', 'Loc'])['Tveg_mean_UAV'].transform('mean')) /
    df.groupby(['Date', 'Loc'])['Tveg_mean_UAV'].transform('std')
)

# Modelling datasets
df_a = df[(df['Date'] == '2024-04-22') | (df['Date'] == '2024-05-14')]
df_v = df[(df['Date'] == '2024-05-29') | (df['Date'] == '2024-06-12')]
df_an = df[(df['Date'] == '2024-04-22') | (df['Date'] == '2024-05-29')]
df_gf = df[(df['Date'] == '2024-05-14') | (df['Date'] == '2024-06-12')]


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
print(df_diff.head())


# In[5]:


def normalize_indexes(df, index_columns):
    data = df.copy()
    for index in index_columns: 
        data[index] = (
        (data[index] - data.groupby(['Date', 'Loc', 'Treatment'])[index].transform('mean')) /
        data.groupby(['Date', 'Loc', 'Treatment'])[index].transform('std')
        )
    return data

df_diff_n = normalize_indexes(df_diff, vi_all)
df_an_n = normalize_indexes(df_an, vi_all)
df_gf_n = normalize_indexes(df_gf, vi_all)
print(df_diff_n)


# In[6]:


# Function to perform feature selection and model evaluation
def feature_selection_rf(df, veg_index_columns, target, min_features=2, max_features=8, direction='backward'):
    print(f"Processing data for dates: {df['Date'].unique()}")
    df = df.dropna()
    training_results = []  
    testing_results = []  
    all_predictions = []   

    X = df[veg_index_columns + ['ETo (mm)', 'water_input_cumulative']]
    y = df[target]

    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

    y = y.values.ravel()  
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if len(X_train) < 5:
        print(f"Not enough samples for cross-validation. Only {len(X_train)} training samples available.")
        return None

    # RandomForestRegressor
    rf = RandomForestRegressor(random_state=16, n_estimators=100)

    # Repeated K-Fold Cross-Validation (5 folds repeated 5 times)
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

    # Sequential Feature Selector
    sfs = SFS(
        rf,
        k_features=(min_features, max_features),
        forward=direction,
        floating=False,
        scoring='neg_mean_absolute_error',
        cv=rkf,
        n_jobs=-1
    )

    print(f"Starting sequential feature selection for target '{target}'...")

    try:
        # Fit the selector on the training data
        sfs.fit(X_train, y_train)

        # Retrieve the best feature combination
        best_features = list(X_train.columns[list(sfs.k_feature_idx_)])
    except Exception as e:
        print(f"Feature selection failed for target '{target}'. Error: {e}")
        return None

    # Train model on selected features
    rf.fit(X_train[best_features], y_train)

    # Training predictions and metrics
    y_train_pred = rf.predict(X_train[best_features])
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    training_results.append({
        'Dates': df['Date'].unique(),
        'Target': target,
        'Selected Features': best_features,
        'Num Features': len(best_features),
        'MAE': train_mae,
        'RMSE': train_rmse,
        'MAPE': train_mape,
        'R2': train_r2,
        'Dataset': 'Training'
    })

    # Testing predictions and metrics
    y_test_pred = rf.predict(X_test[best_features])
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    testing_results.append({
        'Dates': df['Date'].unique(),
        'Target': target,
        'Selected Features': best_features,
        'Num Features': len(best_features),
        'MAE': test_mae,
        'RMSE': test_rmse,
        'MAPE': test_mape,
        'R2': test_r2,
        'Dataset': 'Testing'
    })

    print(f"Done for target '{target}'.\n"
          f"Best features: {best_features},\n"
          f"Training MAE: {train_mae}, Testing MAE: {test_mae},\n"
          f"Training MAPE: {train_mape}, Testing MAPE: {test_mape},\n"
          f"Training RMSE: {train_rmse}, Testing RMSE: {test_rmse},\n"
          f"Training R²: {train_r2}, Testing R²: {test_r2}")

    # Store individual predictions
    train_preds_df = pd.DataFrame({
        'Code': df.loc[X_train.index, 'Code'],
        'Location': df.loc[X_train.index, 'Loc'],
        'Treatment': df.loc[X_train.index, 'Treatment'],
        'Observed Yield': y_train,
        'Predicted Yield': y_train_pred,
        'Dataset': 'Training'
    })

    test_preds_df = pd.DataFrame({
        'Code': df.loc[X_test.index, 'Code'],
        'Location': df.loc[X_test.index, 'Loc'],
        'Treatment': df.loc[X_test.index, 'Treatment'],
        'Observed Yield': y_test,
        'Predicted Yield': y_test_pred,
        'Dataset': 'Testing'
    })

    all_predictions = pd.concat([train_preds_df, test_preds_df], ignore_index=True)

    training_df = pd.DataFrame(training_results)
    testing_df = pd.DataFrame(testing_results)

    return training_df, testing_df, all_predictions


# Feature selection and evaluation for each dataset
train_results_diff, test_results_diff, results_dataset_diff = feature_selection_rf(df_diff_n, vi_all, 'Yield (kg/ha)')
train_results_an, test_results_an, results_dataset_an = feature_selection_rf(df_an_n, vi_all, 'Yield (kg/ha)')
train_results_gf, test_results_gf, results_dataset_gf = feature_selection_rf(df_gf_n, vi_all, 'Yield (kg/ha)')


# In[3]:


os.chdir('results\\feature_selection')


# In[5]:


def save_rf_results(train_results, test_results, results_dataset, name):

    train_results['Dataset'] = 'Training'
    test_results['Dataset'] = 'Testing'

    model_results = pd.concat([train_results, test_results], ignore_index=True)

    prediction_results = pd.merge(results_dataset, model_results[['R2','RMSE','MAE','MAPE', 'Dataset']], on='Dataset', how ='left') 

    data = prediction_results.copy()
    data['Code'] = data['Code'].astype('category')
    data['Loc_Treatment'] = data['Location'] + "_" + data['Treatment']
    data['Loc_Treatment'] = data['Loc_Treatment'].astype('category')

    data = data.rename(columns={'Predicted Yield': 'Predicted_Yield'})

    # Fit a Linear Mixed Model (LMM) with environment random slope
    lmm = MixedLM.from_formula(
        "Predicted_Yield ~ Code",    # Fixed effect: genotype
        groups=data["Loc_Treatment"], # Random effect: environment
        data=data,
        re_formula="1"               # Random slope for environment
    )
    lmm_result = lmm.fit()

    # Extract Estimated Marginal Means (EMMs)
    emm_df = data[['Code']].drop_duplicates().copy()
    emm_df["EMM predicted"] = lmm_result.predict(emm_df)

    # Classify genotypes into yield groups
    upper_quartile = emm_df["EMM predicted"].quantile(0.75)
    lower_quartile = emm_df["EMM predicted"].quantile(0.25)

    def classify_yield(value):
        if value >= upper_quartile:
            return "High yield"
        elif value <= lower_quartile:
            return "Low yield"
        else:
            return "Intermediate yield"

    emm_df["Yield group predicted"] = emm_df["EMM predicted"].apply(classify_yield)

    prediction_results = pd.merge(prediction_results, emm_df, on = 'Code', how='left')

    prediction_results.to_csv(f'yield_predictions_{name}.csv', index=False)

    print(f"Saved: RF_results_{name}_date_site_treatment.csv and yield_predictions_{name}.csv")
    
    train_results = train_results.rename(columns={
        'MAE': 'MAE_train',
        'RMSE': 'RMSE_train',
        'MAPE': 'MAPE_train',
        'R2': 'R2_train'
    })

    test_results = test_results.rename(columns={
        'MAE': 'MAE_test',
        'RMSE': 'RMSE_test',
        'MAPE': 'MAPE_test',
        'R2': 'R2_test'
    })

    train_results['Selected Features'] = train_results['Selected Features'].apply(str)
    test_results['Selected Features'] = test_results['Selected Features'].apply(str)

    final_results = pd.merge(
        train_results[['Selected Features', 'Num Features', 'MAE_train', 'RMSE_train', 'MAPE_train', 'R2_train']],
        test_results[['Selected Features', 'Num Features', 'MAE_test', 'RMSE_test', 'MAPE_test', 'R2_test']],
        on=['Selected Features', 'Num Features'],
        how='inner'
    )

    # Save results
    #final_results.to_csv(f'Yield_results_{name}_date_site_treatment.csv', index=False)

    return prediction_results


pred_results_diff = save_rf_results(train_results_diff, test_results_diff, results_dataset_diff, 'diff')
pred_results_an = save_rf_results(train_results_an, test_results_an, results_dataset_an, 'an')
pred_results_gf = save_rf_results(train_results_gf, test_results_gf, results_dataset_gf, 'gf')


# In[45]:


os.chdir('..')
os.chdir('..')
os.chdir('..')
os.chdir('results\\feature_selection')

# Function to create scatter plots for Observed vs Predicted Yield
def plot_yield_comparison(ax, df, title):
    sns.scatterplot(x=df['Observed Yield'], y=df['Predicted Yield'], hue=df['Dataset'],
                    palette='colorblind', ax=ax)

    ax.plot([df['Observed Yield'].min(), df['Observed Yield'].max()],
            [df['Observed Yield'].min(), df['Observed Yield'].max()],
            color='black', linestyle='--')

    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xlabel('Observed Yield', fontsize=16)
    ax.set_ylabel('Predicted Yield', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    metrics = ['R2', 'RMSE', 'MAE', 'MAPE']
    legend_labels = []

    for dataset in ['Training', 'Testing']:
        subset = df[df['Dataset'] == dataset]
        if not subset.empty:
            errors = {metric: subset[metric].iloc[0] for metric in metrics if metric in subset.columns}
            error_text = ', '.join([f"{'R²' if metric == 'R2' else metric}: {value:.2f}" for metric, value in errors.items()])
            legend_labels.append(f"{dataset} ({error_text})")

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, legend_labels, fontsize=13)

fig, axes = plt.subplots(3, 1, figsize=(8, 24))  
plt.subplots_adjust(hspace=0.35)

VI_selected_an = ['Hue_UAV', 'GA_UAV', 'NDVI_UAV', 'SAVI_UAV', 'RDVI_UAV', 'Hue_ground']
VI_selected_gf = ['v*_UAV', 'Tveg_mean_UAV', 'NDVI_ground', 'a*_ground', 'v*_ground', 'CSI_ground']
VI_selected_diff = ['Lightness_UAV', 'u*_UAV', 'v*_UAV', 'GA_UAV', 'Tveg_mean_UAV', 'NGRDIveg_ground']
vi_lists = [VI_selected_an, VI_selected_gf, VI_selected_diff]

os.chdir('..')
os.chdir('..\\results\\feature_selection')
pred_results_an = pd.read_csv("yield_predictions_an.csv")
pred_results_gf = pd.read_csv("yield_predictions_gf.csv")
pred_results_diff = pd.read_csv("yield_predictions_diff.csv")

os.chdir('..')
os.chdir('..\\results\\figures\\feature_selection')

plot_yield_comparison(axes[0], pred_results_an, 'Anthesis')
plot_yield_comparison(axes[1], pred_results_gf, 'Grain filling')
plot_yield_comparison(axes[2], pred_results_diff, 'Index-difference')

# Add Selected Features under each subplot
for ax, VI_selected in zip(axes, vi_lists):
    pos = ax.get_position()
    x_center = (pos.x0 + pos.x1) / 2
    y_bottom = pos.y0  

    # Two-line split
    half = len(VI_selected) // 2
    vi_text_1 = ', '.join(VI_selected[:half])
    vi_text_2 = ', '.join(VI_selected[half:])

    fig.text(x_center, y_bottom - 0.035, f"Selected features: {vi_text_1}", ha='center', fontsize=16, fontweight='bold')
    fig.text(x_center, y_bottom - 0.045, vi_text_2, ha='center', fontsize=16, fontweight='bold')

plt.savefig("yield_prediction_results_vertical.png", dpi=300, bbox_inches='tight')
plt.show()

