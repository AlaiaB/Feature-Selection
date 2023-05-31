import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json

# Load data
DIR_DATA = "./datos"
df_preoutliers = pd.read_csv(DIR_DATA + '/datos_pre_outliers.csv')

# Load verbose names
with open(DIR_DATA + '/nombres_verbose.json') as f:
    f_verbose = json.load(f)

def a_verbose(nombres, idioma='en'):
    return [f_verbose[idioma][n] for n in nombres]

# Load variables
with open(DIR_DATA + '/var_sets.json') as f:
    var_sets = json.load(f)

# Convert integer columns to float
int_to_dbl = df_preoutliers.select_dtypes(include='int').columns
df_preoutliers[int_to_dbl] = df_preoutliers[int_to_dbl].astype(float)

# Identify numeric and factor variables
nums = list(set(df_preoutliers.select_dtypes(include='float').columns).intersection(set(var_sets["continuas"])))
targets = ['Ventilacion', 'EXITUS', 'UCI', 'TiempoIngreso']
tracks = ['REGISTRO', 'Fingplan', 'Faltplan', 'Fecha_emision', 'Ola']
facts = list(set(var_sets["cte"]) - set(nums))
vars = nums + targets
# Create a dictionary that maps verbose names to original names
verbose_to_original = {a_verbose([var])[0]: var for var in vars}

# Create Dash application
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id='var-dropdown',
            options=[{'label': i, 'value': i} for i in a_verbose(vars)],
            value=a_verbose([vars[0]])[0]
        ),
        dcc.Slider(
            id='n_bins-slider',
            min=2,
            max=100,
            step=1,
            value=30,
            marks={},
            updatemode='drag'
        ),
        dcc.RangeSlider(
            id='range-slider',
            min=0,
            max=100,
            step=0.1,
            value=[0, 100],
            marks={},
            updatemode='drag'
        ),
    ], style={'width': '20%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='dist-plot'),
        html.Div(id='outlier-text')
    ], style={'width': '80%', 'display': 'inline-block'})
])

# Define callback for updating plot
@app.callback(
    Output('dist-plot', 'figure'),
    [Input('var-dropdown', 'value'),
     Input('n_bins-slider', 'value'),
     Input('range-slider', 'value')]
)
def update_dist_plot(var, n_bins, range):
    var = verbose_to_original[var]  # Get the original name
    df = df_preoutliers[[var]].copy()
    var_mn = df[var].min()
    var_mx = df[var].max()
    var_range = var_mx - var_mn
    mn = var_mn + var_range * range[0] / 100
    mx = var_mn + var_range * range[1] / 100
    df = df[(df[var] >= mn) & (df[var] <= mx)]
    fig = go.Figure(data=[go.Histogram(x=df[var], nbinsx=n_bins)])
    fig.update_layout(title_text='Distribution of ' + var)
    return fig

# Define callback for updating outlier text
@app.callback(
    Output('outlier-text', 'children'),
    [Input('var-dropdown', 'value'),
     Input('range-slider', 'value')]
)
def update_outlier_text(var, range):
    var = verbose_to_original[var]  # Get the original name
    df = df_preoutliers[[var]].copy()
    original_nvalues = df[var].count()
    var_mn = df[var].min()
    var_mx = df[var].max()
    var_range = var_mx - var_mn
    mn = var_mn + var_range * range[0] / 100
    mx = var_mn + var_range * range[1] / 100
    df = df[(df[var] >= mn) & (df[var] <= mx)]
    nvalues_now = df[var].count()
    return f"Range: [{mn}, {mx}]. {original_nvalues - nvalues_now} values not in current range."

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
