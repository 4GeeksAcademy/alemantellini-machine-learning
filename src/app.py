from utils import db_connect
engine = db_connect()

# your code here
# IMPORTO LA LIBRERÍA PANDAS
import pandas as pd

# CARGO LOS DATOS
total_data = pd.read_csv('AB_NYC_2019.csv')
total_data

