import pandas as pd
import json

DATA_DIR = "./datos"
filtered_data = pd.read_csv(DATA_DIR+"/datos_pre_aggregation.csv")

with open(DATA_DIR+"/var_sets.json", 'r') as f:
    file_vars = json.load(f)

# Variables for maximum, minimum, mean, and first values
max_vars = ["SaturaMax", "TempMax", "FCMax","TADMax", "TASMax"]
min_vars = ["SaturaMin", "TempMin", "FCMin","TADMin", "TASMin"]
first_vars = list(set(filtered_data.columns) - set(max_vars + min_vars + ['Fecha_emision'] + file_vars['analit_full']))

# Update the original values with max, min, and first within each group
filtered_data[max_vars] = filtered_data.groupby(['REGISTRO', 'Fecha_emision'])[max_vars].transform('max')
filtered_data[min_vars] = filtered_data.groupby(['REGISTRO', 'Fecha_emision'])[min_vars].transform('min')
filtered_data[first_vars] = filtered_data.groupby(['REGISTRO', 'Fingplan'])[first_vars].transform('first')

# Fill NA with known values in the same emission
laboratory = list(set(file_vars['analit_full']).intersection(set(filtered_data.columns)))
filtered_data[laboratory] = filtered_data.groupby(['REGISTRO', 'Fecha_emision'])[laboratory].transform(lambda group: group.bfill().ffill())

filtered_data.to_csv("./datos/datos_pre_outliers.csv")
