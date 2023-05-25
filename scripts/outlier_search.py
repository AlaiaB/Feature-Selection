import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from ipywidgets import interact, widgets

# Load data
df_preoutliers = pd.read_csv("./datos/datos_pre_outliers.csv")

# Convert integer columns to float
int_to_dbl = df_preoutliers.select_dtypes(include='int').columns
df_preoutliers[int_to_dbl] = df_preoutliers[int_to_dbl].astype(float)

# Identify numeric and factor variables
nums = df_preoutliers.select_dtypes(include='float').columns
facts = df_preoutliers.select_dtypes(include='object').columns
targets = ['Ventilacion', 'EXITUS', 'UCI', 'TiempoIngreso']
tracks = ['REGISTRO', 'Fingplan', 'Faltplan', 'Fecha_emision', 'Ola']

# Load verbose names
with open('nombres_verbose.json') as f:
    f_verbose = json.load(f)

def a_verbose(nombres, idioma='en'):
    return [f_verbose[idioma][n] for n in nombres]

# Create interactive plot
@interact
def plot_dist(var=widgets.Dropdown(options=nums, description='Variable'),
              n_bins=widgets.IntSlider(min=2, max=100, step=1, value=30, description='Nº of bins'),
              range=widgets.FloatRangeSlider(value=[0.0, 100.0], min=0, max=100, step=0.1, description='Variable range (%)')):
    df = df_preoutliers[[var]].copy()
    var_mn = df[var].min()
    var_mx = df[var].max()
    var_range = var_mx - var_mn
    mn = var_mn + var_range * range[0] / 100
    mx = var_mn + var_range * range[1] / 100
    df = df[(df[var] >= mn) & (df[var] <= mx)]
    plt.figure(figsize=(10, 6))
    sns.histplot(df[var], bins=n_bins, kde=True)
    plt.title('Distribution of ' + var)
    plt.show()
