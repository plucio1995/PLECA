import pandas as pd
import datetime
import trino

# -----------------------------
# Leer predicciones
# -----------------------------
file_path = "/Users/pedro.lucioglovoapp.com/PycharmProjects/PLECA/Drinks/predicciones_productos_KZ_BERT.csv"
df = pd.read_csv(file_path)
df['predicted_category'] = df['predicted_category'].str.strip().str.lower()

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

# Base date: Últimos 7 días
base_date = (datetime.date.today() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')

# Lista de países a procesar
countries = ['KZ']
all_data = []

# -----------------------------
# Partners with no Drinks
# -----------------------------
for country in countries:
    query_products = f'''
    WITH stores AS (
        SELECT DISTINCT
            order_country_code,
            store_address_id,
            store_name,
            order_city_code
        FROM delta.central_order_descriptors_odp.order_descriptors_v2 od
        WHERE od.order_final_status = 'DeliveredStatus'
          AND od.order_vertical = 'Food'
          AND od.order_country_code IN ('{country}')
          AND od.store_name = 'CASSEROLE PlatformA'
          AND date(od.order_activated_local_at) >= date('{base_date}')
    ),
    max_day_cte AS (
        SELECT DISTINCT CAST(p_snapshot_date AS DATE) AS day
        FROM delta.partner_product_availability_odp.product_availability_v2
        WHERE date(p_snapshot_date)>= date('{base_date}')
    )
    SELECT DISTINCT
        s.order_country_code,
        s.store_name,
        s.order_city_code,
        pa.product_id,
        pa.product_name
    FROM delta.partner_product_availability_odp.product_availability_v2 pa
    INNER JOIN stores s
        ON s.store_address_id = pa.store_address_id,
        max_day_cte
    WHERE CAST(pa.p_snapshot_date AS DATE) IN (SELECT day FROM max_day_cte)
      AND pa.product_is_available = TRUE
      AND pa.product_name IS NOT NULL
    '''
    with trino.dbapi.connect(**conn_details) as conn:
        df_tmp = pd.read_sql_query(query_products, conn)
    all_data.append(df_tmp)

# Unificar los datos extraídos en un solo DataFrame
df_stores = pd.concat(all_data, ignore_index=True)

# Merge with the predicted categories dataframe
df_stores = df_stores.merge(df[['product_id', 'predicted_category']], on='product_id', how='left')

# Filtrar las columnas necesarias para la salida
df_filtered = df_stores[['store_name', 'product_name', 'predicted_category','order_city_code']]

# Mostrar los resultados
print(df_filtered.head())

# -----------------------------
# Exportar los resultados a Google Sheets
# -----------------------------
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os

# Ruta al archivo de credenciales JSON para autenticarse en la API de Google Sheets
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/pedro.lucioglovoapp.com/Documents/05_Otros/02_Cred/mineral-name-431814-h3-568ae02587f4.json"
json_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
print("GOOGLE_APPLICATION_CREDENTIALS:", json_path)

# Definir los alcances necesarios para la autenticación
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name(json_path, scope)
client = gspread.authorize(credentials)

# Abre el archivo de Google Sheets por título
file_title = 'PLECA Drink Analysis'
spreadsheet = client.open(file_title)

# Seleccionar la hoja donde se van a escribir los datos
sheet = spreadsheet.worksheet('test')

# Limpiar la hoja antes de actualizar
sheet.clear()

# Actualizar los datos en la hoja desde la celda A1
sheet.update('A1', [df_filtered.columns.values.tolist()] + df_filtered.values.tolist())

print("Datos exportados a Google Sheets.")
