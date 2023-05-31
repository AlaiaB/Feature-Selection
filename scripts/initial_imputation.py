"""Module for data cleaning and initial feature selection based on semantics and missing data."""

import json
import logging
import numpy as np
import pandas as pd

# Define the path to the JSON file
DIR_DAT = "./datos"
FILE_VARIABLES = DIR_DAT + "/var_sets.json"
logging.basicConfig(filename=DIR_DAT + "/log-seleccion.txt", level=logging.INFO)

# Read the JSON file
with open(FILE_VARIABLES, 'r') as file:
    file_vars = json.load(file)

filtered_data = pd.read_csv(DIR_DAT + "/datos_filtrados.csv", low_memory=False)

# Add auxiliary column
filtered_data['idx_'] = range(1, len(filtered_data) + 1)
initial = filtered_data.columns
logging.info(f"Initial variables: {len(filtered_data.columns)} \n {filtered_data.columns}")
def convert_to_numeric(column):
    """Convert columns to numeric format."""
    column = column.copy()

    try:
        column[column.str.contains('>', na=False)] = (
            column[column.str.contains('>', na=False)].str.replace('>', '').astype(float) + 1e-9
        )
        column[column.str.contains('<', na=False)] = (
            column[column.str.contains('<', na=False)].str.replace('<', '').astype(float) - 1e-9
        )
    except AttributeError:
        pass

    return pd.to_numeric(column, errors='coerce')


def probably_continuous(column, min_prop_num=0.95, min_distinct=22):
    """Determine if a column is likely to be continuous."""
    column = column.dropna()
    if len(column.unique()) < min_distinct:
        return False
    else:
        num_nums = column.apply(lambda x: isinstance(x, (int, float))).sum()
        return (num_nums / len(column)) >= min_prop_num


# Identify columns that are likely to be continuous
continuous = [col for col in filtered_data.columns if probably_continuous(filtered_data[col])]
logging.info(f"Continuous variables: {len(continuous)}. \n {continuous}")

# Convert all the identified continuous columns to numeric format
for col in continuous:
    filtered_data[col] = convert_to_numeric(filtered_data[col])

# Standardize 'Fumador' and 'Evaluacion_del_Riesgo_Nutricional__CONUT_' columns
filtered_data['Fumador'] = filtered_data['Fumador'].replace({
    '1': 'No',
    'Pasivo o Pasiva': 'No',
    'Exfumador o Exfumadora': 'Exfumador/a'
})

# Convert 'Fumador' to ordered categorical type
smoker_categories = pd.CategoricalDtype(categories=['No', 'Exfumador/a', 'Si'], ordered=True)
filtered_data['Fumador'] = filtered_data['Fumador'].astype(smoker_categories)

# Convert 'Evaluacion_del_Riesgo_Nutricional__CONUT_' to lower case and categorical type
filtered_data['Evaluacion_del_Riesgo_Nutricional__CONUT_'] = (
    filtered_data['Evaluacion_del_Riesgo_Nutricional__CONUT_'].str.lower()
)
conut_categories = pd.CategoricalDtype(categories=['normal', 'riesgo lev', 'riesgo mod', 'riesgo gra'])
filtered_data['Evaluacion_del_Riesgo_Nutricional__CONUT_'] = (
    filtered_data['Evaluacion_del_Riesgo_Nutricional__CONUT_'].astype(conut_categories)
)

# Get the intersection of 'meds_full' and the columns in 'filtered_data'
medications = list(set(file_vars['meds_full']).intersection(set(filtered_data.columns)))

# Get the intersection of 'hist' and the columns in 'filtered_data'
comorbidities = list(set(file_vars['hist']).intersection(set(filtered_data.columns)))

# Convert all columns in 'medications' to integer type
filtered_data[medications] = filtered_data[medications].apply(pd.to_numeric, errors='coerce')

# Replace NaN values in 'medications' columns with 0
filtered_data[medications] = filtered_data[medications].fillna(0)

# Exclude 'IMC' from 'comorbidities'
filtered_comorbidities = [c for c in comorbidities if c != 'IMC']

# Replace NaN values in 'filtered_comorbidities' columns with 'No'
filtered_data[filtered_comorbidities] = filtered_data[filtered_comorbidities].fillna('No')

# Selecting initial variables
na_count_patient = filtered_data.groupby(['REGISTRO', 'Fingplan']).apply(lambda x: x.isna().mean())
rest_selection = na_count_patient.loc[:, (na_count_patient.dtypes == np.float64) & (na_count_patient.iloc[0, :] <= 0.30)].columns
selection = list(set(file_vars['cte'] + rest_selection.to_list() + file_vars['track']).intersection(set(filtered_data.columns)))
deleted = list(set(filtered_data.columns) - set(selection))
filtered_data = filtered_data[selection]
logging.info(f"Variables eliminated due to insufficient values: {len(deleted)}")

# Discarding some variables based on their semantics
discard = [
    "CENTRO", "PESO", "Talla", "EDAD", "rangoedad", "GS", "RH", "Flags_hematologia", "Sexo", "Nurg", "comentario",
    "comentario_informe", "tipo", "CAMA", "tcama", "ACUSE_DE_RECIBO_EN_SECCION", "Leucocitos", "Hematies",
    "mascarilla1", "mascarilla2", "gafas1", "fechagafas1", "gafas2", "fechagafas2", "Inf_Mul", "Indice_de_ictericia",
    "Indice_de_lipemia", "Indice_de_Saturacion_de_Hierro", "Filtrado_glomerular_estimado__MDRD4_", "NRBC__V_Absoluto_",
    "X_NRBC", "Luc__V_Absoluto_", "X_Luc", "X_IG", "IG__V_Absoluto_", "Trigliceridos", "VIH_Ac_VIH1_VIH2_y_Ag_P24"
]
discard_dates = list(set([col for col in filtered_data.columns if 'fecha' in col or 'fini' in col]) - set(['Fecha_emision']))
selection = list(set(filtered_data.columns) - set(discard + discard_dates))
logging.info(f'Discarded variables (semantics): {len(discard)} | {len(discard_dates)}')
logging.info(f"Columns after removal: {len(filtered_data.columns)} \n {set(filtered_data.columns)}")
filtered_data = filtered_data[selection]

# Laboratory variables can be obtained by exclusion
file_vars['analit_full'] = list(set(file_vars['analit_full'] + list(set(filtered_data.columns) - set(
    medications + comorbidities + ["REGISTRO", "Fingplan", "Faltplan", "Fecha_emision", "Critico", "Ola", "UCI",
                                      "Ventilacion", "Exitus", "TiempoIngreso"] + file_vars['vitals'] + file_vars['track']))))
#with open(FILE_VARIABLES, 'w') as file:
#    json.dump(file_vars, file, indent=4)

# Consistency check
constants = {
    'EXITUS': 'S',
    'UCI': 'S',
    'Fumador': 'Si',
    'Cardio': 'Si',
    'Pulmonar': 'Si',
    'Diabetes': 'Si',
    'Renal': 'Si',
    'Neuro': 'Si',
    'HTA': 'Si',
    'Onco': 'Si',
    'Ventilacion': 'S'
}

for key in constants.keys():
    inconsistencies = filtered_data.groupby(['REGISTRO', 'Fingplan']).agg({key: 'nunique'}).reset_index()

    inconsistencies = inconsistencies[inconsistencies[key] > 1]
    if inconsistencies.empty:
        continue
    for i in range(len(inconsistencies)):
        reg = inconsistencies.iloc[i]['REGISTRO']
        ing = inconsistencies.iloc[i]['Fingplan']
        filtered_data.loc[(filtered_data['REGISTRO'] == reg) & (filtered_data['Fingplan'] == ing), key] = constants[key]

final = filtered_data.columns
removed = set(initial) - set(final)
logging.info(f"Removed columns: {len(removed)} \n Final columns: {set(final)}")

filtered_data.to_csv("./datos/datos_pre_aggregation.csv")