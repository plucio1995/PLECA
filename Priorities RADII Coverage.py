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

# -----------------------------
# Parámetros iniciales
# -----------------------------
countries = ['KZ']
cities = ['UKK']
#partner = ['Burger King']
#partner_value = partner[0]

# Periodos (usamos solo current period)
start_date_cp = '2024-12-01'
finish_date_cp = '2025-02-28'

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
    FROM delta.central_h3_hexagons_odp.city_h3_hexagons h
    WHERE city_code IN ({cities_str})
),
sessions_data AS (
    SELECT
        ds.h8_hexagon AS hex_hash,
        COUNT(ds.dynamic_session_id) AS Sessions,
        cast(coalesce(count(distinct case when count_ce__order_created > 0  and ov.dynamic_session_id is not NULL then ds.dynamic_session_id else null end), 0) as double) /
        cast(count(distinct ds.dynamic_session_id) as double) as CVR
    FROM hexagons AS hh
    LEFT JOIN delta.customer_behaviour_odp.dynamic_sessions_v1 AS ds 
        ON ds.h8_hexagon = hh.h8_hexagon
    LEFT JOIN delta.customer_behaviour_odp.enriched_backend_event__checkout_order_created_v3 AS ov 
        ON ov.dynamic_session_id = ds.dynamic_session_id
    WHERE 
         ds.country_code IN ({countries_str})
        AND ds.city_code IN ({cities_str})
        AND date(ds.session_start_time_local) BETWEEN date('{start_date_cp}') AND date('{finish_date_cp}')
    GROUP BY 1
),
orders AS (
    SELECT
        hh.h8_hexagon AS hex_hash,
        COUNT(DISTINCT ov.order_id) AS Orders
    FROM delta.customer_behaviour_odp.enriched_backend_event__checkout_order_created_v3 AS od
    LEFT JOIN delta.customer_behaviour_odp.dynamic_sessions_v1 AS se 
        ON od.dynamic_session_id = se.dynamic_session_id
    LEFT JOIN hexagons AS hh 
        ON se.h8_hexagon = hh.h8_hexagon
    LEFT JOIN delta.central_order_descriptors_odp.order_descriptors_v2 AS ov 
        ON od.order_id = ov.order_id
    WHERE 
        ov.order_country_code IN ({countries_str})
        AND ov.order_city_code IN ({cities_str})
        AND date(od.p_creation_date) BETWEEN date('{start_date_cp}') AND date('{finish_date_cp}')
        AND ov.order_final_status = 'DeliveredStatus'
        AND ov.order_parent_relationship_type IS NULL
    GROUP BY 1
)
SELECT 
    DISTINCT 
    h.h3_geom,
    h.h3_8_lat,
    h.h3_8_lon,
    o.hex_hash,
    o.Orders,
    s.Sessions,
    (s.CVR)*100 as CVR
FROM hexagons AS h
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
# Query SQL Stores
# -----------------------------
#query2 = f"""
#SELECT DISTINCT(sa.store_address_id), sa.h3_hash_8, h.h3_8_center_lat as h3_8_lat,
#       h.h3_8_center_lon as h3_8_lon, h.h3_8_polygon_wkt AS h3_geom,
#       COALESCE(store_address_maximum_delivery_distance_meters, store_maximum_delivery_distance) AS store_address_maximum_delivery_distance_meters
#FROM "delta"."partner_stores_odp"."store_addresses_v2" sa
#   LEFT JOIN delta.central_h3_hexagons_odp.city_h3_hexagons h ON h.h3_8_hash = sa.h3_hash_8
#   LEFT JOIN "delta"."partner_stores_odp"."stores_v2" s ON sa.store_id = s.store_id
#WHERE sa.p_end_date IS NULL
#  AND s.p_end_date IS NULL
#  AND s.store_is_enabled = TRUE
#  AND store_address_is_deleted = FALSE
#  AND s.store_name = '{partner_value}'
#  AND s.city_code IN ({cities_str})
#"""
#with trino.dbapi.connect(**conn_details) as conn:
#    store = pd.read_sql_query(query2, conn)
#display(store.head())

# -----------------------------
# Filtrar únicamente Current Period (ya que la query solo devuelve ese periodo)
# -----------------------------
df_current = df.copy()

# -----------------------------
# Creación de Density Buckets
# -----------------------------
def create_density_buckets(df, column_name, num_buckets=7, quantile_threshold=0.95):
    """
    Crea buckets de densidad para una métrica dada en el DataFrame.
    """
    if column_name not in df.columns:
        print(f"Warning: Column '{column_name}' not found in DataFrame. Skipping...")
        return df

    df[column_name] = pd.to_numeric(df[column_name], errors='coerce').fillna(0)
    upper_limit = np.percentile(df[column_name], quantile_threshold * 100)
    filtered_values = df[column_name][df[column_name] <= upper_limit]
    unique_values = filtered_values.nunique()
    if unique_values < num_buckets:
        print(f"Warning: Not enough unique values in '{column_name}' to create {num_buckets} buckets. Reducing bucket count to {unique_values}.")
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
    df[f'{column_name}_bucket'] = pd.cut(df[column_name], bins=bins, labels=labels, include_lowest=True, right=False, ordered=False)
    return df

# Aplicar buckets a las métricas 'Orders' y 'Sessions'
df_buckets = df_current.copy()
metrics = ['Orders', 'Sessions','CVR']
for metric in metrics:
    df_buckets = create_density_buckets(df_buckets, metric)

# -----------------------------
# Creación de columnas numéricas para cada bucket
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

bucket_columns = ['Orders_bucket', 'Sessions_bucket','CVR_bucket']
for col in bucket_columns:
    unique_buckets = df_buckets[col].dropna().unique()
    sorted_buckets = sorted(unique_buckets, key=lambda b: extract_upper_limit(b))
    bucket_mapping = {bucket: rank for rank, bucket in enumerate(sorted_buckets, start=1)}
    new_column = col + "_number"
    df_buckets[new_column] = df_buckets[col].map(bucket_mapping)

df_buckets['Orders_bucket_number'] = df_buckets['Orders_bucket_number'].astype(int)
df_buckets['Sessions_bucket_number'] = df_buckets['Sessions_bucket_number'].astype(int)
df_buckets['CVR_bucket_number'] = df_buckets['CVR_bucket_number'].astype(int)

print(df_buckets[[col for col in df_buckets.columns if "bucket_number" in col]])
print(df_buckets.head())

# -----------------------------
# Mapeo en el mapa con Folium
# -----------------------------
# Filtrar registros sin geometría
df_buckets = df_buckets[df_buckets['h3_geom'].notna()]
df_buckets['h3_geom'] = df_buckets['h3_geom'].astype(str)
df_buckets['geometry'] = df_buckets['h3_geom'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df_buckets, geometry='geometry')
gdf.set_crs(epsg=4326, inplace=True)

import seaborn as sns
def generate_colors(n):
    return sns.color_palette("RdYlGn", n).as_hex()

metrics = ["Orders_bucket", "Sessions_bucket",'CVR_bucket']
centroid_lat = gdf['h3_8_lat'].mean()
centroid_lon = gdf['h3_8_lon'].mean()

for metric in metrics:
    if metric not in gdf.columns:
        continue

    unique_buckets = sorted(gdf[metric].dropna().unique(), key=lambda x: extract_upper_limit(str(x)))
    bucket_limits = sorted([extract_upper_limit(str(b)) for b in unique_buckets if extract_upper_limit(str(b)) is not np.nan])
    if len(bucket_limits) < 2:
        continue
    bucket_limits[-1] += 1

    num_buckets = len(bucket_limits)
    fixed_colors = generate_colors(num_buckets)
    color_map = dict(zip(bucket_limits, fixed_colors))

    gdf[f"{metric}_upper"] = gdf[metric].astype(str).apply(extract_upper_limit)
    gdf[f"{metric}_upper"] = pd.to_numeric(gdf[f"{metric}_upper"], errors="coerce")

    center_lat = gdf["h3_8_lat"].mean()
    center_lon = gdf["h3_8_lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodb positron")

    for _, row in gdf.iterrows():
        value = row[f"{metric}_upper"]

        def get_color(val):
            if np.isnan(val):
                return "#D3D3D3"
            for limit in bucket_limits:
                if val <= limit:
                    return color_map[limit]
            return color_map[bucket_limits[-1]]

        fill_color = get_color(value)
        metric_label = metric.replace("_bucket", "")
        popup_content = (
            f"<b>Hexagon:</b> {row['hex_hash']}<br>"
            f"<b>{metric_label}:</b> {row.get(metric_label, 'N/A'):.2f}<br>"
        )

        # En el style_function se reduce la opacidad de relleno para mayor transparencia (0.3)
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

    # Eliminar registros con coordenadas NaN en store
   # store = store.dropna(subset=['h3_8_lat', 'h3_8_lon'])

    # Agregar iconos de tiendas al mapa
   # for _, row in store.iterrows():
   #     folium.Marker(
   #         location=[row['h3_8_lat'], row['h3_8_lon']],
   #         popup=f"Store ID: {row['store_address_id']}",
   #         icon=folium.Icon(color='blue', icon="shopping-cart", prefix="fa")
   #     ).add_to(m)

    # Agregar círculos con radio de la dirección de la tienda
   # for _, row in store.iterrows():
   #     # Agregar marcador con icono de tienda
   #     folium.Marker(
   #         location=[row['h3_8_lat'], row['h3_8_lon']],
   #         popup=f"Store ID: {row['store_address_id']}",
   #         icon=folium.Icon(color='blue', icon="shopping-cart", prefix="fa")  # Icono de tienda
   #     ).add_to(m)

        # Agregar radio con nuevo esquema de colores
   #     folium.Circle(
   #         location=[row['h3_8_lat'], row['h3_8_lon']],
   #         radius=row['store_address_maximum_delivery_distance_meters'],
   #         color="gray",  # Color del borde más neutro
   #         fill_opacity=0.4,
   #         fill_color="lightblue",  # Relleno azul claro en vez de amarillo
   #     ).add_to(m)

        # Generar leyenda para la métrica
    legend_html = f"""
        <div style="
            position: fixed;
            bottom: 50px;
            left: 50px;
            width: 220px;
            height: auto;
            background-color: white;
            border:2px solid grey;
            z-index:9999;
            font-size:14px;
            padding: 10px;
        ">
            <strong>{metric_label}</strong><br>
    """
    for i in range(len(bucket_limits)):
        legend_html += f"""
            <i style="background: {color_map[bucket_limits[i]]}; width: 20px; height: 20px; display: inline-block;"></i>
            {bucket_limits[i-1] if i > 0 else 0} - {bucket_limits[i]} <br>
        """
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    file_name = f"coverage_{metric_label}.html"
    m.save(file_name)

    # -----------------------------
    # Subir el mapa a Google Drive
    # -----------------------------
    SCOPES = ["https://www.googleapis.com/auth/drive"]
    SERVICE_ACCOUNT_FILE = "/Users/pedro.lucioglovoapp.com/Documents/05_Otros/02_Cred/mineral-name-431814-h3-568ae02587f4.json"
    FOLDER_ID = "1eFXAe26dE2XZuQFtWzSn4j7ifs_cYwrN"

    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    drive_service = build("drive", "v3", credentials=creds)

#    print("Buscando archivos anteriores en la carpeta...")
#    query = f"'{FOLDER_ID}' in parents and mimeType='text/html'"
#    files = drive_service.files().list(q=query, fields="files(id, name)").execute()
#    if "files" in files and len(files["files"]) > 0:
#        for file in files["files"]:
#           try:
#                drive_service.files().delete(fileId=file["id"]).execute()
#                print(f"Archivo eliminado: {file['name']} (ID: {file['id']})")
#            except Exception as e:
#                print(f"No se pudo eliminar {file['name']}: {e}")
#    else:
#        print("No hay archivos anteriores que eliminar.")

    if os.path.exists(file_name):
        print(f"Subiendo {file_name}...")
        file_metadata = {
            "name": file_name,
            "mimeType": "text/html",
            "parents": [FOLDER_ID]
        }
        media = MediaFileUpload(file_name, mimetype="text/html")
        new_file = drive_service.files().create(body=file_metadata, media_body=media, fields="id").execute()
        print(f"Archivo subido: {file_name} con ID: {new_file.get('id')}")
        print(f"URL de acceso: https://drive.google.com/file/d/{new_file.get('id')}/view")
    else:
        print(f"Advertencia: No se encontró {file_name}. Saltando...")

print("Proceso de subida completado.")
