import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objs as go
import pandas as pd
import json

DIR_DATA = "./datos"

def load_data():
    """
    Load CSV and JSON data.
    Return DataFrame, verbose names dictionary, and variables sets dictionary.
    """
    df_preoutliers = pd.read_csv(f"{DIR_DATA}/datos_pre_outliers.csv")
    with open(f"{DIR_DATA}/nombres_verbose.json") as f:
        f_verbose = json.load(f)
    with open(f"{DIR_DATA}/var_sets.json") as f:
        var_sets = json.load(f)
    return df_preoutliers, f_verbose, var_sets

def process_data(df_preoutliers, var_sets):
    """
    Process data to get needed variables.
    Return dictionary mapping verbose names to original names and a list of variables.
    """
    # Convert integer columns to float
    int_cols = df_preoutliers.select_dtypes(include='int').columns
    df_preoutliers[int_cols] = df_preoutliers[int_cols].astype(float)

    # Identify numeric and factor variables
    nums = list(set(df_preoutliers.select_dtypes(include='float').columns).intersection(set(var_sets["continuas"])))
    targets = ['Ventilacion', 'EXITUS', 'UCI', 'TiempoIngreso']
    vars = nums + targets
    verbose_to_original = {a_verbose([var], f_verbose)[0]: var for var in vars}
    
    return verbose_to_original, vars

def a_verbose(names, verbose_dict, language='en'):
    """
    Get verbose names for given names.
    Return a list of verbose names.
    """
    return [verbose_dict[language][n] for n in names]

df_preoutliers, f_verbose, var_sets = load_data()
VERBOSE_TO_ORIGINAL, VARS = process_data(df_preoutliers, var_sets)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id='var-dropdown',
            options=[{'label': i, 'value': i} for i in a_verbose(VARS, f_verbose)],
            value=a_verbose([VARS[0]], f_verbose)[0]
        ),
        dcc.Dropdown(
            id='plot-type-dropdown',
            options=[{'label': i, 'value': i} for i in ['Box Plot', 'Violin Plot', 'Jitter Plot']],
            value='Box Plot'
        ),
        dcc.Slider(
            id='n_bins-slider',
            min=2,
            max=100,
            step=1,
            value=30,
            marks={2:"2", 100:"100"},
            updatemode='drag'
        ),
        dcc.RangeSlider(
            id='range-slider',
            min=0,
            max=100,
            step=0.1,
            value=[0, 100],
            marks={0: '0%', 100: '100%'},
            updatemode='drag'
        ),
    ], style={'width': '20%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='histogram-plot', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='box-violin-plot', style={'width': '50%', 'display': 'inline-block'}),
        html.Div(id='outlier-text'),
        html.Div(id='missing-text')
    ], style={'width': '80%', 'display': 'inline-block'})
])


def get_selected_data(var, range):
    """Extract the required data within the specified range."""
    df = df_preoutliers[[var]].copy()
    var_mn = df[var].min()
    var_mx = df[var].max()
    var_range = var_mx - var_mn
    mn = var_mn + var_range * range[0] / 100
    mx = var_mn + var_range * range[1] / 100
    df = df[(df[var] >= mn) & (df[var] <= mx)]
    return df

@app.callback(
    Output('histogram-plot', 'figure'),
    [Input('var-dropdown', 'value'),
     Input('n_bins-slider', 'value'),
     Input('range-slider', 'value')]
)
def update_histogram_plot(var, n_bins, range):
    """
    Update histogram plot based on user input.
    Return a Figure object.
    """
    var_original = VERBOSE_TO_ORIGINAL[var]
    df = get_selected_data(var_original, range)
    fig = go.Figure(data=[go.Histogram(x=df[var_original], nbinsx=n_bins)])
    fig.update_layout(title_text=f'Histogram of {var}')
    return fig

@app.callback(
    Output('box-violin-plot', 'figure'),
    [Input('var-dropdown', 'value'),
     Input('range-slider', 'value'),
     Input('plot-type-dropdown', 'value')]
)
def update_box_violin_plot(var, range, plot_type):
    """
    Update box/violin plot based on user input.
    Return a Figure object.
    """
    var_original = VERBOSE_TO_ORIGINAL[var]
    df = get_selected_data(var_original, range)
    if plot_type == 'Box Plot':
        fig = go.Figure(data=[go.Box(y=df[var_original], boxpoints='outliers', pointpos=0, line_color='green')])
    elif plot_type == 'Violin Plot':
        fig = go.Figure(data=[go.Violin(y=df[var_original], box_visible=True, line_color='black', meanline_visible=True)])
    elif plot_type == 'Jitter Plot':
        jitter = 0.3  # Adjust this value for your needs
        fig = go.Figure(data=[go.Scatter(x=np.random.uniform(-jitter, jitter, size=len(df[var_original])), y=df[var_original], mode='markers')])
        fig.update_xaxes(range=[-1, 1])
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(range=(df[var_original].min(), df[var_original].max()))
    fig.update_layout(title_text=f'{plot_type} of {var}')
    return fig


@app.callback(
    Output('outlier-text', 'children'),
    [Input('var-dropdown', 'value'),
     Input('range-slider', 'value')]
)
def update_outlier_text(var, range):
    """
    Update outlier text based on user input.
    Return a string.
    """
    var = VERBOSE_TO_ORIGINAL[var]  # Get the original name
    original_nvalues = df_preoutliers[var].count()
    df = get_selected_data(var, range)
    n_values_now = df[var].count()
    return f"Range: [{range[0]}, {range[1]}]. {original_nvalues - n_values_now} values not in current range."

@app.callback(
    Output('missing-text', 'children'),
    [Input('var-dropdown', 'value')]
)
def update_missing_text(var):
    """
    Update text for missing data based on user input.
    Return a string.
    """
    var = VERBOSE_TO_ORIGINAL[var]  # Get the original name
    n_missing = df_preoutliers[var].isnull().sum()
    total = df_preoutliers[var].shape[0]
    missing_perc = (n_missing / total) * 100
    return f"Missing data: {missing_perc:.2f}%"

if __name__ == '__main__':
    app.run_server(debug=True)