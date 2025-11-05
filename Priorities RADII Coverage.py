import pandas as pd
import geopandas as gpd
import folium
from shapely import wkt
from IPython.display import display
import trino
import numpy as np
import re
import matplotlib.colors as mcolors
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os
import seaborn as sns

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
    df_test = pd.read_sql_query('SELECT 1', conn)
display(df_test.head())

# -----------------------------
# Parámetros iniciales
# -----------------------------
countries = ['KZ']
cities = ['NUR']
# Lista de partners a iterar
partners = [ 'Hardee''s']
# Periodos (usamos solo current period)
start_date_cp = '2025-05-01'
finish_date_cp = '2025-07-13'

# Convertir listas en formato SQL
countries_str = ", ".join(f"'{c}'" for c in countries)
cities_str = ", ".join(f"'{c}'" for c in cities)

# -----------------------------
# Query SQL con CTEs: hexagons, orders y sessions para Current Period
# -----------------------------
query1 = f"""
WITH hexagons AS (
    SELECT
        h3_8_hash AS h8_hexagon,
        h3_8_center_lat AS h3_8_lat,
        h3_8_center_lon AS h3_8_lon,
        h3_8_polygon_wkt AS h3_geom
    FROM delta.central_h3_hexagons_odp.city_h3_hexagons
    WHERE city_code IN ({cities_str})
),
sessions_data AS (
    SELECT
        ds.h8_hexagon AS hex_hash,
        COUNT(ds.dynamic_session_id) AS Sessions,
        100.0 * SUM(CASE 
                    WHEN count_ce__order_created > 0 AND ov.dynamic_session_id IS NOT NULL 
                    THEN 1 
                    ELSE 0 
                  END) / COUNT(ds.dynamic_session_id) AS CVR
    FROM delta.customer_behaviour_odp.dynamic_sessions_v1 ds
    INNER JOIN hexagons hh
        ON ds.h8_hexagon = hh.h8_hexagon
    LEFT JOIN delta.customer_behaviour_odp.enriched_backend_event__checkout_order_created_v3 ov 
        ON ds.dynamic_session_id = ov.dynamic_session_id
    WHERE ds.country_code IN ({countries_str})
      AND ds.city_code IN ({cities_str})
      AND ds.session_start_time_local BETWEEN DATE('{start_date_cp}') AND DATE('{finish_date_cp}')
    GROUP BY ds.h8_hexagon
),
orders AS (
    SELECT
        hh.h8_hexagon AS hex_hash,
        COUNT(DISTINCT ov.order_id) AS Orders
    FROM delta.central_order_descriptors_odp.order_descriptors_v2 ov
    INNER JOIN delta.customer_behaviour_odp.enriched_backend_event__checkout_order_created_v3 od  
        ON od.order_id = ov.order_id
    INNER JOIN delta.customer_behaviour_odp.dynamic_sessions_v1 se 
        ON od.dynamic_session_id = se.dynamic_session_id
    INNER JOIN hexagons hh 
        ON se.h8_hexagon = hh.h8_hexagon
    WHERE ov.order_country_code IN ({countries_str})
      AND od.country IN ({countries_str})
      AND od.city IN ({cities_str})
      AND ov.order_city_code IN ({cities_str})
      AND od.p_creation_date BETWEEN DATE('{start_date_cp}') AND DATE('{finish_date_cp}')
      AND ov.order_final_status = 'DeliveredStatus'
      AND ov.order_parent_relationship_type IS NULL
    GROUP BY hh.h8_hexagon
)
SELECT 
    DISTINCT 
    h.h3_geom,
    h.h3_8_lat,
    h.h3_8_lon,
    h.h8_hexagon,
    o.hex_hash,
    o.Orders,
    s.Sessions,
    s.CVR
FROM hexagons h
LEFT JOIN orders o 
    ON h.h8_hexagon = o.hex_hash
LEFT JOIN sessions_data s  
    ON h.h8_hexagon = s.hex_hash
"""

# Recuperar datos usando la query
with trino.dbapi.connect(**conn_details) as conn:
    df = pd.read_sql_query(query1, conn)
display(df.head())


# -----------------------------
# Función para crear Density Buckets
# -----------------------------
def create_density_buckets(df, column_name, num_buckets=7, quantile_threshold=0.95):
    if column_name not in df.columns:
        print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")
        return df

    df[column_name] = pd.to_numeric(df[column_name], errors='coerce').fillna(0)
    upper_limit = np.percentile(df[column_name], quantile_threshold * 100)
    filtered_values = df[column_name][df[column_name] <= upper_limit]
    unique_values = filtered_values.nunique()
    if unique_values < num_buckets:
        print(
            f"Warning: Not enough unique values in '{column_name}' to create {num_buckets} buckets. Reducing bucket count to {unique_values}.")
        num_buckets = max(2, unique_values)
    bins = pd.qcut(filtered_values, q=num_buckets, duplicates='drop', retbins=True)[1]
    bins = np.unique(bins)
    bins = np.sort(bins)
    if len(bins) < 2:
        print(f"Warning: Not enough distinct values in '{column_name}' for binning. Skipping bucket creation.")
        return df
    bins = list(bins)
    bins[-1] = float("inf")
    labels = []
    seen_labels = set()
    for i in range(len(bins) - 1):
        if bins[i + 1] == float("inf"):
            label = f">{int(bins[i])}"
        else:
            label = f"{int(bins[i])} - {int(bins[i + 1])}"
        while label in seen_labels:
            label += "_dup"
        seen_labels.add(label)
        labels.append(label)
    df[f'{column_name}_bucket'] = pd.cut(df[column_name], bins=bins, labels=labels, include_lowest=True, right=False,
                                         ordered=False)
    return df


# Aplicar buckets a la métrica 'Sessions'
df_current = df.copy()
df_current = create_density_buckets(df_current, 'Sessions')


# -----------------------------
# Función para extraer límite superior de bucket
# -----------------------------
def extract_upper_limit(bucket):
    if isinstance(bucket, str):
        match = re.search(r"(\d+)\s*-\s*(\d+)", bucket)
        if match:
            return int(match.group(2))
        match = re.search(r">\s*(\d+)", bucket)
        if match:
            return int(match.group(1)) + 1
    return np.nan


# Crear columna numérica a partir del bucket de Sessions
bucket_columns = ['Sessions_bucket']
for col in bucket_columns:
    unique_buckets = df_current[col].dropna().unique()
    sorted_buckets = sorted(unique_buckets, key=lambda b: extract_upper_limit(b))
    bucket_mapping = {bucket: rank for rank, bucket in enumerate(sorted_buckets, start=1)}
    new_column = col + "_number"
    df_current[new_column] = df_current[col].map(bucket_mapping)

# Si existen columnas de otros buckets (por ejemplo, Orders o CVR) se convierten a int
if 'Orders_bucket_number' in df_current.columns:
    df_current['Orders_bucket_number'] = df_current['Orders_bucket_number'].astype(int)
if 'Sessions_bucket_number' in df_current.columns:
    df_current['Sessions_bucket_number'] = df_current['Sessions_bucket_number'].astype(int)
if 'CVR_bucket_number' in df_current.columns:
    df_current['CVR_bucket_number'] = df_current['CVR_bucket_number'].astype(int)

print(df_current[[col for col in df_current.columns if "bucket_number" in col]])
print(df_current.head())

# -----------------------------
# Crear GeoDataFrame a partir de la geometría
# -----------------------------
df_current = df_current[df_current['h3_geom'].notna()]
df_current['h3_geom'] = df_current['h3_geom'].astype(str)
df_current['geometry'] = df_current['h3_geom'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df_current, geometry='geometry')
gdf.set_crs(epsg=4326, inplace=True)


# -----------------------------
# Función para generar paleta de colores
# -----------------------------
def generate_colors(n):
    return sns.color_palette("RdYlGn", n).as_hex()


# Para la generación del mapa se usará la métrica "Sessions_bucket"
metric = "Sessions_bucket"
unique_buckets = sorted(gdf[metric].dropna().unique(), key=lambda x: extract_upper_limit(str(x)))
bucket_limits = sorted(
    [extract_upper_limit(str(b)) for b in unique_buckets if extract_upper_limit(str(b)) is not np.nan])
if len(bucket_limits) < 2:
    raise Exception("No hay suficientes valores distintos para crear la leyenda de buckets.")
bucket_limits[-1] += 1  # Ajuste para el límite superior

num_buckets = len(bucket_limits)
fixed_colors = generate_colors(num_buckets)
color_map = dict(zip(bucket_limits, fixed_colors))

# Agregar columna con límite superior a cada registro
gdf[f"{metric}_upper"] = gdf[metric].astype(str).apply(extract_upper_limit)
gdf[f"{metric}_upper"] = pd.to_numeric(gdf[f"{metric}_upper"], errors="coerce")


# Función para asignar color según el valor
def get_color(val):
    if np.isnan(val):
        return "#D3D3D3"
    for limit in bucket_limits:
        if val <= limit:
            return color_map[limit]
    return color_map[bucket_limits[-1]]


# -----------------------------
# Credenciales de Google Drive (ya configuradas)
# -----------------------------
SCOPES = ["https://www.googleapis.com/auth/drive"]
SERVICE_ACCOUNT_FILE = "/Users/pedro.lucioglovoapp.com/Documents/05_Otros/02_Cred/mineral-name-431814-h3-568ae02587f4.json"
FOLDER_ID = "1uQIVZfiaXOrD6XN5sSq-09ANogV8vyI2"

# -----------------------------
# Iterar por cada partner para generar y guardar el mapa de coverage
# -----------------------------
for partner_value in partners:
    # Query SQL para obtener las tiendas del partner actual
    query2 = f"""
    SELECT DISTINCT
        sa.store_address_id,
        sa.h3_hash_8,
        h.h3_8_center_lat AS h3_8_lat,
        h.h3_8_center_lon AS h3_8_lon,
        h.h3_8_polygon_wkt AS h3_geom,
        COALESCE(store_address_maximum_delivery_distance_meters, store_maximum_delivery_distance) AS store_address_maximum_delivery_distance_meters
    FROM "delta"."partner_stores_odp"."store_addresses_v2" sa
    LEFT JOIN delta.central_h3_hexagons_odp.city_h3_hexagons h 
        ON h.h3_8_hash = sa.h3_hash_8
    LEFT JOIN "delta"."partner_stores_odp"."stores_v2" s 
        ON sa.store_id = s.store_id
    WHERE sa.p_end_date IS NULL
      AND s.p_end_date IS NULL
      AND s.store_is_enabled = TRUE
      AND store_address_is_deleted = FALSE
      AND s.store_name  = 'Hardee''s'
      AND s.city_code IN ({cities_str})
    """
    with trino.dbapi.connect(**conn_details) as conn:
        store = pd.read_sql_query(query2, conn)
    display(store.head())

    # Crear mapa centrado en la media de los hexágonos
    center_lat = gdf["h3_8_lat"].mean()
    center_lon = gdf["h3_8_lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodb positron")

    # Agregar hexágonos con coverage (basado en Sessions)
    for _, row in gdf.iterrows():
        value = row[f"{metric}_upper"]
        fill_color = get_color(value)
        metric_label = metric.replace("_bucket", "")
        popup_content = (
            f"<b>Hexagon:</b> {row['hex_hash']}<br>"
            f"<b>{metric_label}:</b> {row.get(metric_label, 'N/A')}<br>"
        )

        folium.GeoJson(
            row['geometry'].__geo_interface__,
            style_function=lambda feature, fill_color=fill_color: {
                'fillColor': fill_color,
                'color': "transparent",
                'weight': 0,
                'fillOpacity': 0.6
            },
            popup=folium.Popup(popup_content, max_width=250)
        ).add_to(m)

    # Agregar tiendas del partner actual (asegurarse de que existan las coordenadas)
    store = store.dropna(subset=['h3_8_lat', 'h3_8_lon'])
    for _, row in store.iterrows():
        folium.Marker(
            location=[row['h3_8_lat'], row['h3_8_lon']],
            popup=f"Store ID: {row['store_address_id']}",
            icon=folium.Icon(color='blue', icon="shopping-cart", prefix="fa")
        ).add_to(m)

    # Agregar círculos según el radio de la tienda
    for _, row in store.iterrows():
        folium.Circle(
            location=[row['h3_8_lat'], row['h3_8_lon']],
            radius=row['store_address_maximum_delivery_distance_meters'],
            color="gray",
            fill_opacity=0.4,
            fill_color="lightblue",
        ).add_to(m)

    # Agregar leyenda al mapa
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        width: 220px;
        background-color: white;
        border:2px solid grey;
        z-index:9999;
        font-size:14px;
        padding: 10px;
    ">
        <strong>{metric_label} - {partner_value}</strong><br>
    """
    for i in range(len(bucket_limits)):
        lower = bucket_limits[i - 1] if i > 0 else 0
        legend_html += f"""
        <i style="background: {color_map[bucket_limits[i]]}; width: 20px; height: 20px; display: inline-block;"></i>
        {lower} - {bucket_limits[i]}<br>
        """
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    # Guardar mapa con nombre que incluya la métrica y el partner
    file_name = f"coverage_{metric_label}_{partner_value}.html"
    m.save(file_name)
    print(f"Mapa de coverage para {partner_value} guardado como {file_name}")

    # -----------------------------
    # Subir el mapa a Google Drive
    # -----------------------------
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    drive_service = build("drive", "v3", credentials=creds)

    if os.path.exists(file_name):
        print(f"Subiendo {file_name} a Google Drive...")
        file_metadata = {
            "name": file_name,
            "mimeType": "text/html",
            "parents": [FOLDER_ID]
        }
        media = MediaFileUpload(file_name, mimetype="text/html")
        new_file = drive_service.files().create(
            body=file_metadata, media_body=media, fields="id"
        ).execute()
        print(f"Archivo subido: {file_name} con ID: {new_file.get('id')}")
        print(f"URL de acceso: https://drive.google.com/file/d/{new_file.get('id')}/view")
    else:
        print(f"Advertencia: No se encontró {file_name}. Saltando...")

print("Proceso de generación y subida de mapas completado.")
