#!/usr/bin/env python
# coding: utf-8

# ## Datos Siembra-cosecha ##
# 
# - Aranjuez:
#     - Siembra: 2023-11-22
#     - Cosecha: 
#         - Secano: 2024-07-17
#         - Riego: 2024-07-17
# - Valladolid:
#     - Siembra:
#         - Secano: 2023-12-15
#         - Riego: 2023-12-12
#     - Cosecha: 
#         - Secano: 2024-07-18
#         - Riego: 2024-08-08
#      
# ## Datos Riego ##
# 
# - Aranjuez:
#     - 25/04/2024: 62 mm
#     - 09/05/2024: 93 mm
#     - 16/05/2024: 93 mm
#     - 23/05/2024: 93 mm
#     - 30/05/2024: 93 mm
# - Valladolid:
#     - 19/04/2024
#     - 06/05/2024
#     - 10/05/2024
#     - 31/05/2024
#     - 06/06/2024
#     - all dates 12 mm
# 

# In[1]:


import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import calendar


# In[5]:


os.chdir('C:\\Users\\jaure\\OneDrive - Universitat de Barcelona (1)\\UB\\Doctorat\\Holistic Wheat\\analysis\\datasets')
# os.chdir('C:\\Users\\thoma\\OneDrive - Universitat de Barcelona\\UB\\Doctorat\\Holistic Wheat\\analysis\\datasets')

df = pd.read_csv('holistic_wheat_merged_multi.csv')

df_v = df[df['Loc'] == 'Valladolid']
df_a = df[df['Loc'] == 'Aranjuez']

dates_v =  df_v['Date'].drop_duplicates().sort_values().tolist()
dates_a = df_a['Date'].drop_duplicates().sort_values().tolist()


# In[9]:


# GDD Calculation with Tbase set to 1
os.chdir("C:\\Users\\jaure\\OneDrive - Universitat de Barcelona (1)\\UB\\Doctorat\\Holistic Wheat\\analysis\\datasets\\meteo")
# os.chdir("C:\\Users\\thoma\\OneDrive - Universitat de Barcelona\\UB\\Doctorat\\Holistic Wheat\\analysis\\datasets\\meteo")

# Valladolid
v = pd.read_csv("Finca Zamadueñas_12_12_2023_08_08_2024.csv")
v['Fecha'] = pd.to_datetime(v['Fecha'], format='%Y-%m-%d', errors='coerce')
v = v[(v['Fecha'] >= pd.to_datetime("2023-12-12")) & (v['Fecha'] <= pd.to_datetime("2024-06-12"))]
v = v[['Fecha', 'Temp Max (ºC)', 'Temp Min (ºC)']]
v['Loc'] = 'Valladolid'
v.rename(columns={'Fecha': 'Date'}, inplace=True)

dates_v = pd.to_datetime(dates_v, format='%Y-%m-%d', errors='coerce')

# Aranjuez
a = pd.read_csv("Aranjuez_01_11_2023_12_07_2024.csv")
a['Fecha'] = pd.to_datetime(a['Fecha'], format='%Y-%m-%d', errors='coerce')
a = a[(a['Fecha'] >= pd.to_datetime("2023-11-22")) & (a['Fecha'] <= pd.to_datetime("2024-05-31"))]
a = a[['Fecha', 'Temp Max (ºC)', 'Temp Min (ºC)']]
a['Loc'] = 'Aranjuez'
a.rename(columns={'Fecha': 'Date'}, inplace=True)

def calculate_gdd(tmax, tmin, Tbase):
    # Adjust TMAX and TMIN if they are below TBASE
    tmax = max(tmax, Tbase)
    tmin = max(tmin, Tbase)
    
    # Calculate average temperature
    avg_temp = (tmax + tmin) / 2

    # if avg_temp < Tbase:
    #     avg_temp = Tbase

    gdd = avg_temp - Tbase
    
    return gdd

v['GDD'] = v.apply(lambda row: calculate_gdd(row['Temp Max (ºC)'], row['Temp Min (ºC)'], 0), axis=1)
a['GDD'] = a.apply(lambda row: calculate_gdd(row['Temp Max (ºC)'], row['Temp Min (ºC)'], 0), axis=1)

print(v.tail())
print(a.tail())


# In[11]:


# Function to calculate GDD sum for each location and date
def calculate_gdd_up_to_date(temp_data, unique_dates, location):
    results = []

    for date in unique_dates:
        subset_data = temp_data[temp_data['Date'] <= pd.to_datetime(date)]
        gdd_sum = subset_data['GDD'].sum()
        results.append({
            'Location': location,
            'Date': pd.to_datetime(date),
            'GDD': round(gdd_sum, 2)
        })
    
    gdd_df = pd.DataFrame(results)
    return gdd_df

gdd_valladolid = calculate_gdd_up_to_date(v, dates_v, "Valladolid")
gdd_aranjuez = calculate_gdd_up_to_date(a, dates_a, "Aranjuez")

gdd_df = pd.concat([gdd_valladolid, gdd_aranjuez], ignore_index=True)
print(gdd_df)
os.chdir("C:\\Users\\jaure\\OneDrive - Universitat de Barcelona (1)\\UB\\Doctorat\\Holistic Wheat\\analysis\\datasets\\meteo")
# os.chdir("C:\\Users\\thoma\\OneDrive - Universitat de Barcelona\\UB\\Doctorat\\Holistic Wheat\\analysis\\datasets\\meteo")
gdd_df.to_csv("GDD_dates_all_locations.csv", index=False)


# In[13]:


## Cumulative ETo calculation

os.chdir("C:\\Users\\jaure\\OneDrive - Universitat de Barcelona (1)\\UB\\Doctorat\\Holistic Wheat\\analysis\\datasets\\meteo")
# os.chdir("C:\\Users\\thoma\\OneDrive - Universitat de Barcelona\\UB\\Doctorat\\Holistic Wheat\\analysis\\datasets\\meteo")

# Valladolid
v = pd.read_csv("Finca Zamadueñas_12_12_2023_08_08_2024.csv")
v['Fecha'] = pd.to_datetime(v['Fecha'], format='%Y-%m-%d', errors='coerce')
v = v[(v['Fecha'] >= pd.to_datetime("2023-12-12")) & (v['Fecha'] <= pd.to_datetime("2024-06-12"))]
v = v[['Fecha', 'ETo (mm)', 'Precipitation (mm)']]
v['ETo_cumulative'] = v['ETo (mm)'].cumsum()
v['prec_cumulative'] = v['Precipitation (mm)'].cumsum()
v['Loc'] = 'Valladolid'
v.rename(columns={'Fecha': 'Date'}, inplace=True)

dates_v = pd.to_datetime(dates_v, format='%Y-%m-%d', errors='coerce')
#v = v[v['Date'].isin(dates_v)]

# Aranjuez
a = pd.read_csv("Aranjuez_01_11_2023_12_07_2024.csv")
a['Fecha'] = pd.to_datetime(a['Fecha'], format='%Y-%m-%d', errors='coerce')
a = a[(a['Fecha'] >= pd.to_datetime("2023-11-22")) & (a['Fecha'] <= pd.to_datetime("2024-05-31"))]
a = a[['Fecha', 'ETo (mm)', 'Precipitation (mm)']]
a['ETo_cumulative'] = a['ETo (mm)'].cumsum()
a['prec_cumulative'] = a['Precipitation (mm)'].cumsum()
a['Loc'] = 'Aranjuez'
a.rename(columns={'Fecha': 'Date'}, inplace=True)

dates_a = pd.to_datetime(dates_a, format='%Y-%m-%d', errors='coerce')
#a = a[a['Date'].isin(dates_a)]


Et0_prec_df = pd.concat([v, a], ignore_index=True)
print(a)


# In[15]:


# Irrigation data

# Corrected data for irrigation
data_irr = {
    "Loc": ["Aranjuez"] * 5 + ["Valladolid"] * 5,  
    "Date": [
        "2024-04-25", "2024-05-09", "2024-05-16", 
        "2024-05-23", "2024-05-30", "2024-04-19", 
        "2024-05-06", "2024-05-10", "2024-05-31", 
        "2024-06-06"
    ],
    "Irrigation": [
        62, 93, 93, 93, 93, 12, 12, 12, 12, 12
    ]
}

df_irr = pd.DataFrame(data_irr)

df_irr["Date"] = pd.to_datetime(df_irr["Date"])

print(df_irr)


# In[17]:


Et0_prec_df['Irrigation'] = 0
for _, row in df_irr.iterrows():
    mask = (Et0_prec_df['Loc'] == row['Loc']) & (Et0_prec_df['Date'] == row['Date'])
    Et0_prec_df.loc[mask, 'Irrigation'] += row['Irrigation']

missing_rows = df_irr[~df_irr['Date'].isin(Et0_prec_df['Date'])]
if not missing_rows.empty:
    missing_rows = missing_rows.assign(
        **{
            'ETo (mm)': 0,
            'Precipitation (mm)': 0,
            'ETo_cumulative': None,
            'prec_cumulative': None,
        }
    )
    Et0_prec_df = pd.concat([Et0_prec_df, missing_rows], ignore_index=True)

Et0_prec_df.sort_values(by=['Loc', 'Date'], inplace=True)

Et0_prec_df['water_input_cumulative'] = 0

# Calculate cumulative water input location-wise
for loc in Et0_prec_df['Loc'].unique():
    loc_mask = Et0_prec_df['Loc'] == loc
    Et0_prec_df.loc[loc_mask, 'water_input_cumulative'] = (
        Et0_prec_df.loc[loc_mask, 'prec_cumulative'].fillna(method='ffill').fillna(0) +
        Et0_prec_df.loc[loc_mask, 'Irrigation'].cumsum()
    )

print(Et0_prec_df)
#Et0_prec_df.to_csv("ETo_prec_all_locations.csv", index=False)

## This is the cumulative water input for the supplementary irrigation, modify dataset so that rainfed only has cumulative rainfall as water input ##


# In[83]:


os.chdir("C:\\Users\\jaure\\OneDrive - Universitat de Barcelona (1)\\UB\\Doctorat\\Holistic Wheat\\analysis\\datasets\\meteo")
# os.chdir("C:\\Users\\thoma\\OneDrive - Universitat de Barcelona\\UB\\Doctorat\\Holistic Wheat\\analysis\\datasets\\meteo")

df = pd.read_csv("Final_meteo_dataset.csv")

df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
df = df.dropna(subset=['Fecha'])

df['MonthNum'] = df['Fecha'].dt.month.astype(int)
df['MonthName'] = df['MonthNum'].apply(lambda x: calendar.month_abbr[x])

monthly_data = df.groupby(['Location', 'MonthNum', 'MonthName']).agg({
    'EtPMon': 'sum',
    'Rainfed (mm)': 'sum',
    'Irrigated (mm)': 'sum'
}).reset_index()

month_order = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
monthly_data['MonthName'] = pd.Categorical(monthly_data['MonthName'], categories=month_order, ordered=True)
monthly_data = monthly_data.sort_values(by='MonthName')

locations = monthly_data['Location'].unique()
fig, axes = plt.subplots(nrows=1, ncols=len(locations), figsize=(16, 6), sharey=True)

for ax, location in zip(axes, locations):
    loc_data = monthly_data[monthly_data['Location'] == location]
    
    ax.bar(loc_data['MonthName'], loc_data['EtPMon'], color='#FDB863', label='ETo (mm)')
    
    ax.plot(loc_data['MonthName'], loc_data['Rainfed (mm)'], marker='o', color='blue', label='Rainfed (mm)')
    
    ax.plot(loc_data['MonthName'], loc_data['Irrigated (mm)'], marker='s', color='green', label='Irrigated (mm)')
    
    ax.set_title(f'{location}', fontsize=18)
    ax.tick_params(axis='x', rotation=45, labelsize=13)

fig.text(0.5, -0.005, 'Month (2023 - 2024)', ha='center', fontsize=18)
axes[0].set_ylabel('Cumulative monthly values (mm)', fontsize=16)

axes[0].legend(loc='upper left', fontsize=13)
plt.tight_layout()
#plt.subplots_adjust(bottom=0.18)
os.chdir("C:\\Users\\jaure\\OneDrive - Universitat de Barcelona (1)\\UB\\Doctorat\\Holistic Wheat\\analysis\\results\\figures")
# os.chdir("C:\\Users\\thoma\\OneDrive - Universitat de Barcelona\\UB\\Doctorat\\Holistic Wheat\\analysis\\results\\figures")
plt.savefig('weather_plot.png', dpi=300, bbox_inches='tight')
plt.show()

