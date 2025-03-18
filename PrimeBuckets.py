import pandas as pd
import trino
from sklearn.cluster import KMeans

## Starburst credentials (this works alone, no need to modify)
HOST = 'starburst.g8s-data-platform-prod.glovoint.com'
PORT = 443
conn_details = {
    'host': HOST,
    'port': PORT,
    'http_scheme': 'https',
    'auth': trino.auth.OAuth2Authentication()
}

# 1. Ejecutar la query principal para obtener las métricas por cliente, incluyendo PLTV
countries = ['KZ']
all_data = []

for country in countries:
    query_cities = f'''
    WITH prime_orders AS (
    SELECT
        od.order_country_code as country,
        od.order_id,
        od.order_transacted_value_eur as aov,
        cm.contribution_margin_eur as contribution_margin_eur
    FROM delta.central_order_descriptors_odp.order_descriptors_v2 od
    LEFT JOIN delta.finance_financial_reports_odp.pnl_order_level cm ON od.order_id = cm.order_id
    WHERE
        date_trunc('year', od.order_activated_local_at) >= date_trunc('year', date_add('year', -1, date(current_timestamp)))
        AND od.order_country_code = '{country}'
        AND od.order_final_status = 'DeliveredStatus'
        AND od.order_parent_relationship_type IS NULL
        AND od.order_is_prime=true
        AND od.order_vertical='QCommerce'
        AND od.store_name='SMALL'
)
SELECT * FROM prime_orders
    '''
    with trino.dbapi.connect(**conn_details) as conn:
        df = pd.read_sql_query(query_cities, conn)
    all_data.append(df)

df_cities = pd.concat(all_data, ignore_index=True)
print("Datos de clientes:")
print(df_cities.head())

# 2. Aplicar clustering (K-means) por país en base a AOV
num_clusters = 4
bucket_names = {0: '1. Low', 1: '2. Medium Low', 2: '3. Medium High', 3: '4. High'}
df_cities['bucket'] = None

for country in df_cities['country'].unique():
    subset = df_cities[df_cities['country'] == country].copy()
    subset = subset.dropna(subset=['aov'])
    if len(subset) < num_clusters:
        df_cities.loc[subset.index, 'bucket'] = 'Low'
        continue

    # Filtrar outliers (IQR)
    Q1 = subset['aov'].quantile(0.25)
    Q3 = subset['aov'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Aplicamos clustering solo a los valores dentro del rango
    subset_filtered = subset[(subset['aov'] >= lower_bound) & (subset['aov'] <= upper_bound)]
    if len(subset_filtered) < num_clusters:
        subset_filtered = subset  # Usamos todos los datos si hay pocos puntos para clusterizar

    n_clusters_country = min(num_clusters, len(subset_filtered))
    kmeans = KMeans(n_clusters=n_clusters_country, random_state=42)
    clusters = kmeans.fit_predict(subset_filtered[['aov']])
    subset_filtered['cluster'] = clusters

    # Ordenamos los clusters por su mediana de AOV
    cluster_medians = subset_filtered.groupby('cluster')['aov'].median()
    cluster_order = cluster_medians.sort_values().index.tolist()
    mapping = {original: bucket_names[i] for i, original in enumerate(cluster_order)}

    df_cities.loc[subset_filtered.index, 'bucket'] = subset_filtered['cluster'].map(mapping)

    # Asignar valores por encima del upper_bound al bucket 'High'
    df_cities.loc[(df_cities['country'] == country) & (df_cities['aov'] > upper_bound), 'bucket'] = '4. High'

# 3. Calcular métricas por grupo (país y bucket) utilizando solo clientes con datos de PLTV
group_metrics = df_cities.groupby(['country', 'bucket']).agg(
    total_orders=('order_id', 'count'),
    contribution_margin_eur=('contribution_margin_eur', 'sum'),
    aov_min=('aov', 'min'),
    aov_max=('aov', 'max')
).reset_index()

# Agregar columna para mostrar el intervalo de AOV (ej. "100 - 200")
group_metrics['aov_interval'] = [
    f"{int(min_val)} - {int(max_val)}"
    for min_val, max_val in zip(group_metrics['aov_min'], group_metrics['aov_max'])
]

# Calcular la proporción de orders por bucket respecto al total de orders por país
group_metrics['orders_ratio'] = group_metrics['total_orders'] / group_metrics.groupby('country')['total_orders'].transform('sum')

# Calcular otras métricas
group_metrics['cm'] = group_metrics.apply(
    lambda row: row['contribution_margin_eur'] / row['total_orders'] if row['total_orders'] != 0 else 0, axis=1
)

print("Métricas por grupo (calculadas solo con clientes con datos de PLTV):")
print(group_metrics)


# 6. Exportar los resultados a Google Sheets
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/pedro.lucioglovoapp.com/Documents/05_Otros/02_Cred/mineral-name-431814-h3-568ae02587f4.json"
json_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
print("GOOGLE_APPLICATION_CREDENTIALS:", json_path)

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name(json_path, scope)
client = gspread.authorize(credentials)

file_title = 'KZ KG - Prime AOV Buckets'
spreadsheet = client.open(file_title)
sheet = spreadsheet.worksheet('data1')

data = [group_metrics.columns.tolist()] + group_metrics.values.tolist()
sheet.clear()
sheet.update('A1', data)
print("Datos exportados a Google Sheets.")
