import pandas as pd
import datetime
import trino
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import re
from unidecode import unidecode
import math

# -----------------------------
# Leer predicciones
# -----------------------------
file_path = "/Users/pedro.lucioglovoapp.com/PycharmProjects/PLECA/Drinks/test products_UA_8.csv"
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

# Base date: últimos 90 días (modifica según lo que necesites)
base_date = (datetime.date.today() - datetime.timedelta(days=90)).strftime('%Y-%m-%d')

# Lista de países a procesar
countries = ['UA']
all_orders = []

# -----------------------------
# Extraer órdenes por país
# -----------------------------
for country in countries:
    query_orders = f'''
    WITH top_partners AS (
      SELECT
          store_name,
          COUNT(DISTINCT order_id) AS orders
      FROM delta.central_order_descriptors_odp.order_descriptors_v2
      WHERE
          DATE(order_activated_local_at) >= DATE('{base_date}')
          AND order_country_code = '{country}'
          AND order_final_status = 'DeliveredStatus'
          AND order_vertical = 'Food'
          AND order_parent_relationship_type IS NULL
          AND store_name IS NOT NULL
      GROUP BY store_name
      ORDER BY orders DESC
      LIMIT 100
    )
    SELECT
          od.order_id,
          od.order_city_code,
          od.order_country_code,
          od.store_name,
          bp.product_id,
          bp.product_name,
          bp.bought_product_quantity,
          od.order_transacted_value_local,
          od.order_transacted_value_eur,
          CASE WHEN tp.store_name IS NOT NULL THEN 'Top_100' END AS Top_partner
      FROM
          "delta"."central_order_descriptors_odp"."order_descriptors_v2" od
      LEFT JOIN
          delta.customer_bought_products_odp.bought_products_v2 bp
          ON bp.order_id = od.order_id
      LEFT JOIN
          top_partners tp
          ON od.store_name = tp.store_name
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

# -----------------------------
# Hacer JOIN con categorías predichas
# -----------------------------
df_orders = df_orders.merge(df[['product_id', 'predicted_category']], on='product_id', how='left')

# Filtrar órdenes donde la categoría está definida (puedes quitar este filtro si deseas trabajar con todas)
df_orders = df_orders[df_orders['predicted_category'].notna() & (df_orders['predicted_category'] != '')]

# Filtrar solo productos clasificados como Drinks
df_drinks_orders = df_orders[df_orders['predicted_category'].isin(['drink', 'drinks'])]

# -----------------------------
# Cálculo de métricas globales por país
# -----------------------------
results_per_country = {}

for country in countries:
    df_country = df_orders[df_orders['order_country_code'] == country]
    df_country_drinks = df_drinks_orders[df_drinks_orders['order_country_code'] == country]

    num_orders = df_country['order_id'].nunique()
    num_orders_with_drinks = df_country_drinks['order_id'].nunique()

    # AOV para todas las órdenes (agrupamos por order_id para garantizar órdenes únicas)
    aov_local = df_country.groupby('order_id')['order_transacted_value_local'].first().mean()
    aov_eur = df_country.groupby('order_id')['order_transacted_value_eur'].first().mean()

    # AOV para órdenes con drinks
    aov_local_with_drinks = df_country_drinks.groupby('order_id')['order_transacted_value_local'].first().mean()
    aov_eur_with_drinks = df_country_drinks.groupby('order_id')['order_transacted_value_eur'].first().mean()

    # Órdenes sin drinks
    df_country_no_drinks = df_country[~df_country['order_id'].isin(df_country_drinks['order_id'])]
    aov_local_no_drinks = df_country_no_drinks.groupby('order_id')['order_transacted_value_local'].first().mean()
    aov_eur_no_drinks = df_country_no_drinks.groupby('order_id')['order_transacted_value_eur'].first().mean()

    # Productos sin categoría predicha o vacía
    missing_or_empty_categories_count = df_country[
        df_country['predicted_category'].isna() | (df_country['predicted_category'] == '')
    ].shape[0]
    products_with_category_count = df_country[
        df_country['predicted_category'].notna() & (df_country['predicted_category'] != '')
    ].shape[0]

    # % de órdenes con drinks que tienen solo productos drink
    orders_with_drinks_ids = df_country_drinks['order_id'].unique()
    exclusively_drink_orders = df_country.groupby('order_id')['predicted_category'].apply(
        lambda cats: all(cat in ['drink', 'drinks'] for cat in cats if pd.notna(cat))
    )
    exclusively_drink_orders = exclusively_drink_orders[exclusively_drink_orders.index.isin(orders_with_drinks_ids)]
    num_exclusively_drink_orders = exclusively_drink_orders.sum()
    total_orders_with_drinks = len(orders_with_drinks_ids)
    percent_exclusively_drink = (num_exclusively_drink_orders / total_orders_with_drinks * 100) if total_orders_with_drinks > 0 else 0

    results_per_country[country] = {
        "Orders": num_orders,
        "Orders with Drinks": num_orders_with_drinks,
        "AOV_local Orders": round(aov_local, 2) if pd.notna(aov_local) else None,
        "AOV_eur Orders": round(aov_eur, 2) if pd.notna(aov_eur) else None,
        "AOV_local Orders Drinks": round(aov_local_with_drinks, 2) if pd.notna(aov_local_with_drinks) else None,
        "AOV_eur Orders Drinks": round(aov_eur_with_drinks, 2) if pd.notna(aov_eur_with_drinks) else None,
        "AOV_local Orders No Drinks": round(aov_local_no_drinks, 2) if pd.notna(aov_local_no_drinks) else None,
        "AOV_eur Orders No Drinks": round(aov_eur_no_drinks, 2) if pd.notna(aov_eur_no_drinks) else None,
        "Productos sin categoría predicha o vacía": missing_or_empty_categories_count,
        "Productos con categoría predicha": products_with_category_count,
        "% Partners with Drinks": None,  # Se asigna más adelante
        "% Exclusively Drinks Orders": round(percent_exclusively_drink, 2)
    }

# -----------------------------
# Top 100 partners: obtener todos, tengan orders con drinks o no
# -----------------------------
top_partners_list = df_orders[df_orders['Top_partner'].notna()]['store_name'].unique()

# Total de órdenes para cada top partner
orders_total_by_top_partner = (
    df_orders[df_orders['store_name'].isin(top_partners_list)]
    .groupby('store_name')['order_id']
    .nunique()
    .reset_index()
    .rename(columns={'order_id': 'total_orders'})
)

# Órdenes con drinks para cada top partner
orders_with_drinks = (
    df_drinks_orders[df_drinks_orders['store_name'].isin(top_partners_list)]
    .groupby('store_name')['order_id']
    .nunique()
    .reset_index()
    .rename(columns={'order_id': 'orders_with_drinks'})
)

# Crear DataFrame con todos los top partners y unir métricas de órdenes
df_top_partners = pd.DataFrame({'store_name': top_partners_list})
df_top_partners = df_top_partners.merge(orders_with_drinks, on='store_name', how='left')
df_top_partners['orders_with_drinks'] = df_top_partners['orders_with_drinks'].fillna(0)
df_top_partners = df_top_partners.merge(orders_total_by_top_partner, on='store_name', how='left')
df_top_partners = df_top_partners.sort_values(by='orders_with_drinks', ascending=False)

orders_by_top_partner = df_top_partners.merge(orders_total_by_top_partner, on='store_name', how='left')

# -----------------------------
# Top partners sin productos Coca-Cola
# -----------------------------
mask_coca = df_orders['product_name'].str.contains(
    r'(coca|koka|kokakola|კოკა|კოკაკოლა|კოლა|кока|кола|կոկա|կոկակոլա|կոլա|Cola|Coca)',
    case=False, na=False, regex=True
)
mask_pepsi = df_orders['product_name'].str.contains(
    r'(pepsi|პეპსი|пепси|փեփսի|пепсi|Պեպսი|Пепсi)',
    case=False, na=False, regex=True
)
coca_partners_1 = df_orders[mask_coca]['store_name'].unique()
pepsi_partners = df_orders[mask_pepsi]['store_name'].unique()
coca_partners = set(coca_partners_1) | set(pepsi_partners)
top_partners_no_coca = [partner for partner in top_partners_list if partner not in coca_partners]

# Calcular número de órdenes para cada partner en top_no_coca
df_top_no_coca_orders = (
    df_orders[df_orders['store_name'].isin(top_partners_no_coca)]
    .groupby('store_name')['order_id']
    .nunique()
    .reset_index()
    .rename(columns={'order_id': 'orders_count'})
)
top_no_coca_data = [["Store Name", "Orders Count"]] + df_top_no_coca_orders.values.tolist()

# -----------------------------
# Partners (tiendas) sin Drinks
# -----------------------------
all_data = []
for country in countries:
    query_products = f'''
    WITH top_partners AS (
      SELECT
          store_name,
          COUNT(DISTINCT order_id) AS orders
      FROM delta.central_order_descriptors_odp.order_descriptors_v2
      WHERE
          DATE(order_activated_local_at) >= DATE('{base_date}')
          AND order_country_code = '{country}'
          AND order_final_status = 'DeliveredStatus'
          AND order_vertical = 'Food'
          AND order_parent_relationship_type IS NULL
          AND store_name IS NOT NULL
      GROUP BY store_name
      ORDER BY orders DESC
      LIMIT 100
    ),
    stores AS (
        SELECT
            order_country_code,
            order_city_code,
            store_address_id,
            od.store_name
        FROM delta.central_order_descriptors_odp.order_descriptors_v2 od
        LEFT JOIN top_partners tp ON tp.store_name = od.store_name
        WHERE od.order_final_status = 'DeliveredStatus'
          AND od.order_vertical = 'Food'
          AND od.order_country_code = '{country}'
          AND DATE(od.order_activated_local_at) >= DATE('{base_date}')
    ),
    product_availability_filtered AS (
        SELECT
            store_address_id,
            product_id,
            product_name,
            product_is_available,
            CAST(p_snapshot_date AS DATE) AS snapshot_date
        FROM delta.partner_product_availability_odp.product_availability_v2
        WHERE DATE(p_snapshot_date) >= DATE('{base_date}')
    ),
    max_days AS (
        SELECT DISTINCT snapshot_date
        FROM product_availability_filtered
    )
    SELECT DISTINCT
        s.order_country_code,
        s.order_city_code,
        s.store_name,
        paf.product_id
    FROM product_availability_filtered paf
    JOIN stores s
        ON s.store_address_id = paf.store_address_id
    JOIN max_days md
        ON paf.snapshot_date = md.snapshot_date
    WHERE paf.product_is_available = TRUE
      AND paf.product_name IS NOT NULL
    '''
    with trino.dbapi.connect(**conn_details) as conn:
        df_tmp = pd.read_sql_query(query_products, conn)
    all_data.append(df_tmp)

df_stores = pd.concat(all_data, ignore_index=True)
df_stores = df_stores.merge(df[['product_id', 'predicted_category']], on='product_id', how='left')
tiendas_drinks = df_stores.groupby('store_name')['predicted_category'].apply(
    lambda x: x.isin(['drink', 'drinks']).any()
)
tiendas_drinks = tiendas_drinks[tiendas_drinks].index.tolist()
df_stores_unique = df_stores[['order_country_code', 'order_city_code', 'store_name']].drop_duplicates()

# Calcular número de órdenes por partner y city
partner_orders = (
    df_orders
    .groupby(['order_country_code', 'order_city_code', 'store_name'])['order_id']
    .nunique()
    .reset_index()
    .rename(columns={'order_id': 'orders_count'})
)
df_stores_unique = df_stores_unique.merge(partner_orders, on=['order_country_code', 'order_city_code', 'store_name'], how='left')
df_stores_no_drinks = df_stores_unique[~df_stores_unique['store_name'].isin(tiendas_drinks)]

# -----------------------------
# Top 10 drinks (nombres normalizados)
# -----------------------------
def normalize_product_name(name):
    name = unidecode(name)
    name = name.lower().strip()
    name = re.sub(r'\b\d+\s?(ml|l|g|gr)\b', '', name)
    name = re.sub(r'[-,\d\.\(\)\'"\s]', '', name)
    name = re.sub(r'\s+', ' ', name)
    name = name.strip()
    if name == "kokakola":
        name = "cocacola"
    return name

df_drinks_orders['normalized_product_name'] = df_drinks_orders['product_name'].apply(normalize_product_name)
top10_drinks = (
    df_drinks_orders
    .groupby('normalized_product_name')['bought_product_quantity']
    .sum()
    .sort_values(ascending=False)
    .head(50)
    .reset_index()
)
top10_data = [list(top10_drinks.columns)] + top10_drinks.values.tolist()

# -----------------------------
# Cálculo del AOV en EUR por partner: total, con drinks y sin drinks
# -----------------------------
# Agrupar órdenes únicas para evitar duplicados (agrupamos por store_name y order_id)
df_orders_unique = df_orders.groupby(['store_name', 'order_id'])['order_transacted_value_eur'].first().reset_index()
df_drinks_orders_unique = df_drinks_orders.groupby(['store_name', 'order_id'])['order_transacted_value_eur'].first().reset_index()

# AOV total por partner
aov_eur_all_by_partner = (
    df_orders_unique
    .groupby('store_name')['order_transacted_value_eur']
    .mean()
    .reset_index()
    .rename(columns={'order_transacted_value_eur': 'AOV_eur_all'})
)

# AOV para órdenes con drinks
aov_eur_drinks_by_partner = (
    df_drinks_orders_unique
    .groupby('store_name')['order_transacted_value_eur']
    .mean()
    .reset_index()
    .rename(columns={'order_transacted_value_eur': 'AOV_eur_drinks'})
)

# AOV para órdenes sin drinks (filtramos aquellas órdenes que no aparecen en drinks)
order_ids_with_drinks = df_drinks_orders['order_id'].unique()
df_orders_no_drinks = df_orders[~df_orders['order_id'].isin(order_ids_with_drinks)]
df_orders_no_drinks_unique = df_orders_no_drinks.groupby(['store_name', 'order_id'])['order_transacted_value_eur'].first().reset_index()
aov_eur_no_drinks_by_partner = (
    df_orders_no_drinks_unique
    .groupby('store_name')['order_transacted_value_eur']
    .mean()
    .reset_index()
    .rename(columns={'order_transacted_value_eur': 'AOV_eur_no_drinks'})
)

# Unir estos cálculos al DataFrame de top partners
df_top_partners = df_top_partners.merge(aov_eur_drinks_by_partner, on='store_name', how='left')
df_top_partners = df_top_partners.merge(aov_eur_no_drinks_by_partner, on='store_name', how='left')
df_top_partners = df_top_partners.merge(aov_eur_all_by_partner, on='store_name', how='left')
df_top_partners['AOV_eur_drinks'] = df_top_partners['AOV_eur_drinks'].round(2)
df_top_partners['AOV_eur_no_drinks'] = df_top_partners['AOV_eur_no_drinks'].round(2)
df_top_partners['AOV_eur_all'] = df_top_partners['AOV_eur_all'].round(2)

# Preparar el DataFrame final para la pestaña df_top50_drinks
df_top50_drinks = df_top_partners.copy()
top50_data = [list(df_top50_drinks.columns)] + df_top50_drinks.values.tolist()

# -----------------------------
# Funciones para sanitizar datos (evitar NaN, Infinity, etc.)
# -----------------------------
def sanitize_cell(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value

def sanitize_data(data):
    return [[sanitize_cell(cell) for cell in row] for row in data]

sanitized_export_data = sanitize_data([["Country", "Total Orders", "Orders with Drinks", "AOV_local Orders",
                "AOV_eur Orders", "AOV_local Orders Drinks", "AOV_eur Orders Drinks",
                "AOV_local Orders No Drinks", "AOV_eur Orders No Drinks",
                "Productos sin categoría predicha o vacía", "Productos con categoría predicha",
                "% Partners with Drinks", "% Exclusively Drinks Orders"]] +
                [[
                    country,
                    metrics["Orders"],
                    metrics["Orders with Drinks"],
                    metrics["AOV_local Orders"],
                    metrics["AOV_eur Orders"],
                    metrics["AOV_local Orders Drinks"],
                    metrics["AOV_eur Orders Drinks"],
                    metrics["AOV_local Orders No Drinks"],
                    metrics["AOV_eur Orders No Drinks"],
                    metrics["Productos sin categoría predicha o vacía"],
                    metrics["Productos con categoría predicha"],
                    metrics["% Partners with Drinks"],
                    metrics["% Exclusively Drinks Orders"]
                ] for country, metrics in results_per_country.items()])

sanitized_stores_data = sanitize_data([["Country", "City", "Store Name", "Orders"]] +
                                df_stores_no_drinks[['order_country_code', 'order_city_code', 'store_name', 'orders_count']].drop_duplicates().values.tolist())
sanitized_top10_data = sanitize_data(top10_data)
sanitized_top50_data = sanitize_data(top50_data)
sanitized_top_no_coca_data = sanitize_data(top_no_coca_data)

# -----------------------------
# Configurar Google Sheets y exportar los datos
# -----------------------------
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/pedro.lucioglovoapp.com/Documents/05_Otros/02_Cred/mineral-name-431814-h3-568ae02587f4.json"
json_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name(json_path, scope)
client = gspread.authorize(credentials)

file_title = 'PLECA Drinks Analysis'
spreadsheet = client.open(file_title)

# Exportar hoja data_
try:
    sheet = spreadsheet.worksheet('data_UA')
    sheet.clear()
except gspread.exceptions.WorksheetNotFound:
    sheet = spreadsheet.add_worksheet(title='data_UA', rows="100", cols="20")
sheet.update(sanitized_export_data, 'A1')
print("Datos exportados a Google Sheets (hoja data).")

# Exportar hoja data_partners_
try:
    sheet_partners = spreadsheet.worksheet('data_partners_UA')
    sheet_partners.clear()
except gspread.exceptions.WorksheetNotFound:
    sheet_partners = spreadsheet.add_worksheet(title='data_partners_UA', rows="100", cols="20")
sheet_partners.update(sanitized_stores_data, 'A1')
print("Datos de las tiendas sin drinks exportados a la hoja data_partners_.")

# Exportar hoja top10_drinks_
try:
    sheet_top10 = spreadsheet.worksheet('top10_drinks_UA')
    sheet_top10.clear()
except gspread.exceptions.WorksheetNotFound:
    sheet_top10 = spreadsheet.add_worksheet(title='top10_drinks_UA', rows="100", cols="20")
sheet_top10.update(sanitized_top10_data, 'A1')
print("Datos del Top 10 drinks exportados a Google Sheets (hoja top10_drinks_).")

# Exportar hoja df_top50_drinks
try:
    sheet_top50_drinks = spreadsheet.worksheet('df_top50_drinks')
    sheet_top50_drinks.clear()
except gspread.exceptions.WorksheetNotFound:
    sheet_top50_drinks = spreadsheet.add_worksheet(title='df_top50_drinks', rows="100", cols="20")
sheet_top50_drinks.update(sanitized_top50_data, 'A1')
print("Datos del Top 50 partners orders with drinks (con AOV) exportados a Google Sheets (hoja df_top50_drinks).")

# Exportar hoja top_no_coca_
try:
    sheet_top_no_coca = spreadsheet.worksheet('top_no_coca_UA')
    sheet_top_no_coca.clear()
except gspread.exceptions.WorksheetNotFound:
    sheet_top_no_coca = spreadsheet.add_worksheet(title='top_no_coca_UA', rows="100", cols="20")
sheet_top_no_coca.update(sanitized_top_no_coca_data, 'A1')
print("Datos de Top partners sin productos Coca-Cola exportados a Google Sheets (hoja top_no_coca_).")

# -----------------------------
# Extraer product_id sin categoría predicha y exportar CSV
# -----------------------------
df_no_category = df_orders[
    df_orders['predicted_category'].isna() | (df_orders['predicted_category'] == '')
][['product_id']].drop_duplicates().head(100)
csv_filename = "product_without_category_AM.csv"
df_no_category.to_csv(csv_filename, index=False)
print(f"Se han exportado {len(df_no_category)} product_id a {csv_filename}.")

# -----------------------------
# Mostrar AOV para órdenes sin drinks
# -----------------------------
print("\nAOV para órdenes sin drinks:")
for country in countries:
    print(f"{country}: AOV Local: {results_per_country[country]['AOV_local Orders No Drinks']}, "
          f"AOV EUR: {results_per_country[country]['AOV_eur Orders No Drinks']}")
