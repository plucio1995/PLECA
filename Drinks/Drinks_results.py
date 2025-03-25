import pandas as pd
import datetime
import trino
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os

# -----------------------------
# Leer predicciones
# -----------------------------
file_path = "/Users/pedro.lucioglovoapp.com/PycharmProjects/PLECA/Drinks/predicciones_productos_KG_4_MultilingualBERT.csv"
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

# Base date: Últimos 30 días (puedes ajustar el período si lo requieres)
base_date = (datetime.date.today() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')

# Lista de países a procesar
countries = ['KG']
all_orders = []

# -----------------------------
# Extraer órdenes por país
# -----------------------------
for country in countries:
    query_orders = f'''
    SELECT
        od.order_id,
        od.order_country_code,
        bp.product_id,
        bp.product_name,
        bp.bought_product_quantity,
        od.order_transacted_value_local,
        od.order_transacted_value_eur
    FROM
        "delta"."central_order_descriptors_odp"."order_descriptors_v2" od
    LEFT JOIN
        delta.customer_bought_products_odp.bought_products_v2 bp
        ON bp.order_id = od.order_id
    WHERE
        od.order_activated_local_at >= DATE('{base_date}')
        AND od.order_final_status = 'DeliveredStatus'
        AND od.order_parent_relationship_type IS NULL
        AND od.order_country_code = '{country}'
        AND od.order_vertical = 'Food'
    '''
    with trino.dbapi.connect(**conn_details) as conn:
        df1 = pd.read_sql_query(query_orders, conn)
    all_orders.append(df1)

# Combinar todas las órdenes
df_orders = pd.concat(all_orders, ignore_index=True)

# Hacer JOIN con categorías
df_orders = df_orders.merge(df[['product_id', 'predicted_category']], on='product_id', how='left')

# Filtrar solo productos clasificados como Drinks
df_drinks_orders = df_orders[df_orders['predicted_category'].isin(['drink', 'drinks'])]

# Diccionario para almacenar resultados por país
results_per_country = {}

# Calcular métricas por país y almacenarlas en el diccionario
for country in countries:
    df_country = df_orders[df_orders['order_country_code'] == country]
    df_country_drinks = df_drinks_orders[df_drinks_orders['order_country_code'] == country]

    num_orders = df_country['order_id'].nunique()
    num_orders_with_drinks = df_country_drinks['order_id'].nunique()

    aov_local = df_country.groupby('order_id')['order_transacted_value_local'].first().mean()
    aov_eur = df_country.groupby('order_id')['order_transacted_value_eur'].first().mean()

    aov_local_with_drinks = df_country_drinks.groupby('order_id')['order_transacted_value_local'].first().mean()
    aov_eur_with_drinks = df_country_drinks.groupby('order_id')['order_transacted_value_eur'].first().mean()

    # Total de productos sin categoría predicha o vacía
    missing_or_empty_categories_count = df_country[
        df_country['predicted_category'].isna() | (df_country['predicted_category'] == '')
    ].shape[0]
    products_with_category_count = df_country[
        df_country['predicted_category'].notna() & (df_country['predicted_category'] != '')
    ].shape[0]

    # Inicialmente asignamos las métricas sin la métrica de partners
    results_per_country[country] = {
        "Orders": num_orders,
        "Orders with Drinks": num_orders_with_drinks,
        "AOV_local Orders": round(aov_local, 2) if pd.notna(aov_local) else None,
        "AOV_eur Orders": round(aov_eur, 2) if pd.notna(aov_eur) else None,
        "AOV_local Orders Drinks": round(aov_local_with_drinks, 2) if pd.notna(aov_local_with_drinks) else None,
        "AOV_eur Orders Drinks": round(aov_eur_with_drinks, 2) if pd.notna(aov_eur_with_drinks) else None,
        "Productos sin categoría predicha o vacía": missing_or_empty_categories_count,
        "Productos con categoría predicha": products_with_category_count
    }

# -----------------------------
# Partners (tiendas) sin Drinks
# -----------------------------
all_data = []
for country in countries:
    query_products = f'''
    WITH stores AS (
        SELECT DISTINCT
            order_country_code,
            order_city_code,
            store_address_id,
            store_name
        FROM delta.central_order_descriptors_odp.order_descriptors_v2 od
        WHERE od.order_final_status = 'DeliveredStatus'
          AND od.order_vertical = 'Food'
          AND od.order_country_code IN ('{country}')
          AND date(od.order_activated_local_at) >= date('{base_date}')
    ),
    max_day_cte AS (
        SELECT DISTINCT CAST(p_snapshot_date AS DATE) AS day
        FROM delta.partner_product_availability_odp.product_availability_v2
        WHERE date(p_snapshot_date) >= date('{base_date}')
    )
    SELECT DISTINCT
        s.order_country_code,
        s.order_city_code,
        s.store_name,
        pa.product_id
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

print("Muestra de datos extraídos:")
print(df_tmp.head())

# Unificar los datos extraídos en un solo DataFrame
df_stores = pd.concat(all_data, ignore_index=True)
print("Datos extraídos de Starburst:")
print(df_stores.head())

# Merge con las categorías predichas
df_stores = df_stores.merge(df[['product_id', 'predicted_category']], on='product_id', how='left')

# Determinar por tienda si tiene al menos un producto drink
tiendas_drinks = df_stores.groupby('store_name')['predicted_category'].apply(
    lambda x: x.isin(['drink', 'drinks']).any()
)
tiendas_drinks = tiendas_drinks[tiendas_drinks].index.tolist()

# Filtrar las tiendas que NO tienen drinks
stores_no_drinks = df_stores[~df_stores['store_name'].isin(tiendas_drinks)]
print("Tiendas sin bebidas en el menú (con país):")
print(stores_no_drinks[['order_country_code', 'store_name']].drop_duplicates())

# Calcular el porcentaje de partners con drinks por país
pct_partners_per_country = {}
for country in countries:
    df_stores_country = df_stores[df_stores['order_country_code'] == country]
    total_partners = df_stores_country['store_name'].nunique()
    partners_with_drinks = df_stores_country[
        df_stores_country['store_name'].isin(tiendas_drinks)
    ]['store_name'].nunique()
    pct = (partners_with_drinks / total_partners * 100) if total_partners > 0 else None
    pct_partners_per_country[country] = round(pct, 2) if pct is not None else None

# Actualizar el diccionario results_per_country para incluir la nueva métrica
for country in countries:
    results_per_country[country]["% Partners with Drinks"] = pct_partners_per_country.get(country, None)

# Imprimir resultados para ver la integración de la nueva métrica
for country, metrics in results_per_country.items():
    print(f"\nResultados para {country}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

# -----------------------------
# Preparar y exportar los resultados a Google Sheets
# -----------------------------
# Definir la cabecera con la nueva métrica
export_data = [["Country", "Total Orders", "Orders with Drinks", "AOV_local Orders",
                "AOV_eur Orders", "AOV_local Orders Drinks", "AOV_eur Orders Drinks",
                "Productos sin categoría predicha o vacía", "Productos con categoría predicha",
                "% Partners with Drinks"]]

for country, metrics in results_per_country.items():
    export_data.append([
        country,
        metrics["Orders"],
        metrics["Orders with Drinks"],
        metrics["AOV_local Orders"],
        metrics["AOV_eur Orders"],
        metrics["AOV_local Orders Drinks"],
        metrics["AOV_eur Orders Drinks"],
        metrics["Productos sin categoría predicha o vacía"],
        metrics["Productos con categoría predicha"],
        metrics["% Partners with Drinks"]
    ])

# Configurar Google Sheets
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/pedro.lucioglovoapp.com/Documents/05_Otros/02_Cred/mineral-name-431814-h3-568ae02587f4.json"
json_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
print("GOOGLE_APPLICATION_CREDENTIALS:", json_path)

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name(json_path, scope)
client = gspread.authorize(credentials)

# Abrir el archivo de Google Sheets
file_title = 'PLECA Drink Analysis'
spreadsheet = client.open(file_title)

# Exportar datos de órdenes a la hoja "data_UA"
sheet = spreadsheet.worksheet('data_KG_2')
sheet.clear()
sheet.update('A1', export_data)
print("Datos exportados a Google Sheets (hoja data_KG).")

# Exportar datos de tiendas sin drinks a la hoja "data_partners_UA"
stores_data = [["Country", "City", "Store Name"]]
for _, row in stores_no_drinks[['order_country_code', 'order_city_code', 'store_name']].drop_duplicates().iterrows():
    stores_data.append([row['order_country_code'], row['order_city_code'], row['store_name']])

sheet_partners = spreadsheet.worksheet('data_partners_KG_2')
sheet_partners.clear()
sheet_partners.update('A1', stores_data)
print("Datos de las tiendas sin drinks exportados a la hoja 'data_partners_UA'.")
