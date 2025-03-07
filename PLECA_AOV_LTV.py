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
countries = ['PL', 'UA', 'KZ', 'KG', 'GE', 'AM']
all_data = []

for country in countries:
    query_cities = f'''
    WITH ncs as (select Distinct customer_id
    from delta.central_order_descriptors_odp.order_descriptors_v2 as od
        where 1=1
        and od.order_country_code = '{country}'
        and date_trunc('year', order_activated_local_at) >= date_trunc('year', date_add('year', -1, date(current_timestamp)))
        and od.order_final_status = 'DeliveredStatus'
        and order_parent_relationship_type is null    
        and order_is_first_delivered_order=true   
    )select
        od.order_country_code as country,
        od.customer_id,
        ltv.pltv_1y,
        ltv.pltv_3y,
        ltv.pltv_5y,
        lto.predicted_orders_12mo as plto_1y,
        lto.predicted_orders_36mo as plto_3y,
        lto.predicted_orders_60mo as plto_5y,
        avg(od.order_transacted_value_eur) as aov,
        count(distinct od.order_id) as total_orders,
        count(distinct CASE WHEN od.order_vertical='QCommerce' THEN od.order_id END) as total_qcomm_orders,
        date_diff(
            'month', 
            min(case when od.order_is_first_delivered_order then od.order_activated_local_at end), 
            max(od.order_activated_local_at)
        ) + 1 as active_months,
        sum(cm.contribution_margin_eur) as contribution_margin_eur,
        SUM(cm.tcorev_eur + cm.adve_eur + cm.totbdr_eur + cm.bwsrev_eur + 
            cm.mbsrsu_eur + cm.serfig_eur + cm.deacsu_eur + (MFCGSO_eur - QMCOGS_eur)) AS RPO,
SUM(COALESCE(bcomfr_eur,0)
+ COALESCE(dicofr_eur,0)
+ COALESCE(wacofr_eur,0)
+ COALESCE(bundfr_eur,0)
+ COALESCE(rubcfr_eur,0)
+ COALESCE(rabcfr_eur,0)
+ COALESCE(otbcfr_eur,0)
+ COALESCE(caocfr_eur,0)) as CPO
    from delta.central_order_descriptors_odp.order_descriptors_v2 as od
    INNER JOIN ncs n ON n.customer_id = od.customer_id
    LEFT JOIN delta.finance_financial_reports_odp.pnl_order_level cm ON od.order_id = cm.order_id
    LEFT JOIN "delta"."growth__ltv_targets_slices__odp"."ltv_targets_customer_level" ltv  ON ltv.customer_id = od.customer_id
    LEFT JOIN "delta"."growth__customer_lto__odp"."customer_lto" lto ON lto.customer_id = od.customer_id
    where 1=1
        and od.order_country_code = '{country}'
        and date_trunc('year', order_activated_local_at) >= date_trunc('year', date_add('year', -1, date(current_timestamp)))
        and od.order_final_status = 'DeliveredStatus'
        and order_parent_relationship_type is null
        and date(lto.p_date)=date(DATE_ADD('day', -1, CURRENT_DATE))
    group by 1,2,3,4,5,6,7,8
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
    cluster_customers=('customer_id', 'count'),
    total_orders=('total_orders', 'sum'),
    total_qcomm_orders=('total_qcomm_orders', 'sum'),
    active_months=('active_months', 'sum'),
    contribution_margin_eur=('contribution_margin_eur', 'sum'),
    RPO_total=('RPO', 'sum'),
    CPO_total=('CPO', 'sum'),
    pltv_1y_sum=('pltv_1y', 'sum'),
    pltv_3y_sum=('pltv_3y', 'sum'),
    pltv_5y_sum=('pltv_5y', 'sum'),
    pltv_1y_count=('pltv_1y', lambda x: x.notnull().sum()),
    pltv_3y_count=('pltv_3y', lambda x: x.notnull().sum()),
    pltv_5y_count=('pltv_5y', lambda x: x.notnull().sum()),
    plto_1y_sum=('plto_1y', 'sum'),
    plto_3y_sum=('plto_3y', 'sum'),
    plto_5y_sum=('plto_5y', 'sum'),
    plto_1y_count=('plto_1y', lambda x: x.notnull().sum()),
    plto_3y_count=('plto_3y', lambda x: x.notnull().sum()),
    plto_5y_count=('plto_5y', lambda x: x.notnull().sum()),
    aov_min=('aov', 'min'),
    aov_max=('aov', 'max')
).reset_index()

# Agregar columna para mostrar el intervalo de AOV (ej. "100 - 200")
group_metrics['aov_interval'] = group_metrics.apply(
    lambda row: f"{int(row['aov_min'])} - {int(row['aov_max'])}", axis=1
)

# Calcular la proporción de orders por bucket respecto al total de orders por país
group_metrics['orders_ratio'] = group_metrics['total_orders'] / group_metrics.groupby('country')['total_orders'].transform('sum')

# Calcular la proporción de customers por bucket respecto al total de customers por país
group_metrics['customers_ratio'] = group_metrics['cluster_customers'] / group_metrics.groupby('country')['cluster_customers'].transform('sum')

# Calcular la proporción de qcomm orders por bucket respecto al total de orders por país
group_metrics['qcomm_ratio'] = group_metrics['total_qcomm_orders'] / group_metrics.groupby('country')['total_orders'].transform('sum')


# Calcular el promedio de cada PLTV utilizando solo los clientes con datos
group_metrics['pltv_1y_avg'] = group_metrics['pltv_1y_sum'] / group_metrics['pltv_1y_count']
group_metrics['pltv_3y_avg'] = group_metrics['pltv_3y_sum'] / group_metrics['pltv_3y_count']
group_metrics['pltv_5y_avg'] = group_metrics['pltv_5y_sum'] / group_metrics['pltv_5y_count']

# Calcular el promedio de cada PLTV utilizando solo los clientes con datos
group_metrics['plto_1y_avg'] = group_metrics['plto_1y_sum'] / group_metrics['plto_1y_count']
group_metrics['plto_3y_avg'] = group_metrics['plto_3y_sum'] / group_metrics['plto_3y_count']
group_metrics['plto_5y_avg'] = group_metrics['plto_5y_sum'] / group_metrics['plto_5y_count']


# Calcular otras métricas
group_metrics['frequency'] = group_metrics.apply(
    lambda row: row['total_orders'] / row['active_months'] if row['active_months'] != 0 else 0, axis=1
)
group_metrics['cm'] = group_metrics.apply(
    lambda row: row['contribution_margin_eur'] / row['total_orders'] if row['total_orders'] != 0 else 0, axis=1
)
group_metrics['RPO'] = group_metrics.apply(
    lambda row: row['RPO_total'] / row['total_orders'] if row['total_orders'] != 0 else 0, axis=1
)
group_metrics['CPO'] = group_metrics.apply(
    lambda row: row['CPO_total'] / row['total_orders'] if row['total_orders'] != 0 else 0, axis=1
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

file_title = 'PLECA AOV buckets'
spreadsheet = client.open(file_title)
sheet = spreadsheet.worksheet('data2')

data = [group_metrics.columns.tolist()] + group_metrics.values.tolist()
sheet.clear()
sheet.update('A1', data)
print("Datos exportados a Google Sheets.")
