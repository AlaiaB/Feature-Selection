import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging


# Read data
DIR_DAT = ".\datos"
F_DAT = DIR_DAT + "\DATOS_COVID19_4_H_V9-10_1_ANONIMIZADO.csv"
datos_orig = pd.read_csv(F_DAT, na_values=[np.nan, "", "----", "Muestra he", "(!)"], low_memory=False)

# Set up logging
logging.basicConfig(filename=DIR_DAT + "/log-limpieza.txt", level=logging.INFO)

def log_pacientes(msg):
    logging.info(f"{msg} Samples: {datos_filt1.shape[0]}, Patients: {datos_filt1['REGISTRO'].nunique()}")

# Convert to datetime
for col in ['Fecha_emision', 'Fingplan', 'Faltplan']:
    datos_orig[col] = pd.to_datetime(datos_orig[col])

# Filter patients who have been admitted and discharged or passed away
mask = (~datos_orig['Faltplan'].isna() | (datos_orig['Faltplan'].isna() & datos_orig['EXITUS'] == "S")) & (~datos_orig['Fingplan'].isna() & ~datos_orig['REGISTRO'].isna())
datos_filt1 = datos_orig.loc[mask]
log_pacientes("After filtering for admitted patient:")


# Filter patients older than 18
datos_filt1 = datos_filt1.loc[datos_filt1['Edad'] >= 18]
log_pacientes("After age filtering:")

# Clean dates
logging.info(f"Samples with multiple admissions for the same discharge and vice versa: {datos_filt1.groupby(['REGISTRO', 'Faltplan'])['Fingplan'].nunique().loc[lambda x: x > 1].count()}")
logging.info(f"Samples with multiple discharges for the same admission: {datos_filt1.groupby(['REGISTRO', 'Fingplan'])['Faltplan'].nunique().loc[lambda x: x > 1].count()}")
datos_filt1['Fingplan'] = datos_filt1.groupby(['REGISTRO', 'Faltplan'])['Fingplan'].transform('min')
datos_filt1['Faltplan'] = datos_filt1.groupby(['REGISTRO', 'Fingplan'])['Faltplan'].transform('max')


# Remove patients with admission date later than discharge date
logging.info(f"Patients with admission date after discharge date: {datos_filt1.loc[datos_filt1['Fingplan'] > datos_filt1['Faltplan']]['REGISTRO'].nunique()}")
logging.info(f"Samples with admission date after discharge date: {datos_filt1.loc[datos_filt1['Fingplan'] > datos_filt1['Faltplan']].shape[0]}")
datos_filt1 = datos_filt1.loc[datos_filt1['Fingplan'] <= datos_filt1['Faltplan']]
log_pacientes("After admission and discharge date filtering:")

# Filter patients with at least one positive PCR test
datos_filt1['PCR'] = datos_filt1['Deteccion_del_nuevo_Coronavirus__COVID_19__SARS_2__mediante_PCR'].apply(lambda x: 'positivo' if 'positiv' in str(x).lower() else ('negativo' if 'negativ' in str(x).lower() else np.nan))
datos_filt1 = datos_filt1.groupby(['REGISTRO', 'Fingplan']).filter(lambda x: (x['PCR'] == 'positivo').sum() >= 1)
log_pacientes("After accepting only admissions with at least one positive PCR:")

# Filter samples within 2 days before admission and 7 days after
N_DIAS_POS = 7
N_DIAS_PREV = 2
mask = (datos_filt1['Fecha_emision'] <= datos_filt1['Fingplan'] + timedelta(days=N_DIAS_POS)) & (datos_filt1['Fecha_emision'] >= datos_filt1['Fingplan'] - timedelta(days=N_DIAS_PREV))
datos_filt1 = datos_filt1.loc[mask]
log_pacientes(f"Collected samples {N_DIAS_PREV} days before and {N_DIAS_POS} days after admission:")

# Compute 'TiempoIngreso'
datos_filt1['TiempoIngreso'] = (datos_filt1['Faltplan'] - datos_filt1['Fingplan']).dt.days

# Use only the last admission of each patient
ing_pac = datos_filt1.groupby('REGISTRO')['Fingplan'].max().reset_index()
datos_filt1 = pd.merge(datos_filt1, ing_pac, on=['REGISTRO', 'Fingplan'])
log_pacientes("After only using the last admission of each patient:")

# Assign 'Ola' based on 'Fingplan'
FICHERO_INTERVALOS = DIR_DAT + "/intervalos_olas.json"
with open(FICHERO_INTERVALOS) as f:
    f_intervalos = json.load(f)

def asignar_ola(fecha):
    for ola, interval in f_intervalos.items():
        if pd.to_datetime(interval[0]) <= fecha <= pd.to_datetime(interval[1]):
            return ola
    return np.nan

datos_filt1['Ola'] = datos_filt1['Fingplan'].apply(asignar_ola)
logging.info(f"Samples by wave: {datos_filt1['Ola'].value_counts().to_dict()}")
logging.info(f"Patients by wave: {datos_filt1.groupby('Ola')['REGISTRO'].nunique().to_dict()}")

# Replace missing 'VACUNA' values
datos_filt1.loc[datos_filt1['Ola'].isin([1, 2, 3, 4]), 'VACUNA'] = "Valor ausente"
datos_filt1['VACUNA'].fillna("Valor ausente", inplace=True)

datos_filt1.to_csv("./datos/datos_filtrados.csv")
