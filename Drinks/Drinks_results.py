import trino
import pandas as pd
import numpy as np
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from IPython.display import display
import pandas as pd
import geopandas as gpd
import folium
from shapely import wkt
from IPython.display import display
import trino
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import re
import matplotlib.colors as mcolors
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os
import datetime


# -----------------------------
# Configuración de conexión a Starburst
# -----------------------------
HOST = 'starburst.g8s-data-platform-prod.glovoint.com'
PORT = 443
conn_details = {
    'host': HOST,
    'port': PORT,
    'http_scheme': 'https',
    'auth': trino.auth.OAuth2Authentication()
}
# Prueba de conexión
with trino.dbapi.connect(**conn_details) as conn:
    df_test = pd.read_sql_query('select 1', conn)
display(df_test.head())

# Lista de países a procesar
base_date = (datetime.date.today() - datetime.timedelta(weeks=1)).strftime('%Y-%m-%d')
countries = ['PL', 'UA', 'KZ']
all_data = []

print(base_date)

query1= f"""
        SELECT DISTINCT
            order_country_code,
            store_address_id,
            store_name
        FROM delta.central_order_descriptors_odp.order_descriptors_v2 od
        WHERE od.order_final_status = 'DeliveredStatus'
          AND od.order_vertical = 'Food'
          AND od.order_country_code ='KZ'
          AND date(od.order_activated_local_at) >= date('{base_date}')
   """
# Recuperar datos usando la query
with trino.dbapi.connect(**conn_details) as conn:
    df = pd.read_sql_query(query1, conn)
display(df.head())