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
import datetime

#############################################
# 1. Conexión a Starburst y extracción de productos
#############################################
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
base_date = (datetime.date.today() - datetime.timedelta(days=90)).strftime('%Y-%m-%d')
countries = ['KZ']
all_data = []

for country in countries:
    query_products = f'''
    WITH stores AS (
        SELECT DISTINCT
            order_country_code,
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
        s.store_name,
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

print(df_tmp.head())

# Unificar los datos extraídos en un solo DataFrame
df_products = pd.concat(all_data, ignore_index=True)
print("Datos extraídos de Starburst:")
print(df_products.head())

# Mostrar la distribución de países en df_products
print("\nDistribución de order_country_code en df_products:")
print(df_products['order_country_code'].value_counts())

#############################################
# 2. Carga del dataset de entrenamiento y creación de modelos por país
#############################################
# Cargar el dataset de entrenamiento (debe contener las columnas 'order_country_code', 'product_name' y 'Category')
df_train = pd.read_csv(
    "/Users/pedro.lucioglovoapp.com/PycharmProjects/PLECA/Drinks/PLECA orders with Drinks - Drinks Total.csv")
print("\nDataset de entrenamiento:")
print(df_train.head())

# Mostrar la distribución de países en df_train
print("\nDistribución de order_country_code en df_train:")
print(df_train['order_country_code'].value_counts())

# Cargar spaCy y definir función para extraer embeddings de texto
nlp = spacy.load('en_core_web_md')  # Modelo preentrenado con embeddings


def get_embeddings(text):
    doc = nlp(text)
    return doc.vector


# Crear modelos por país utilizando únicamente 'product_name'
models_by_country = {}
# Filtrar filas con valores nulos en product_name o Category
df_train = df_train.dropna(subset=['product_name', 'Category'])
grouped_train = df_train.groupby('order_country_code')

for country, group in grouped_train:
    print(f"\nEntrenando modelo para el país: {country}")
    # Eliminar nulos en product_name y Category (ya se hizo globalmente, pero se refuerza)
    group = group.dropna(subset=['product_name', 'Category'])
    if group.empty:
        print(f"  - No hay datos en el grupo para {country}.")
        continue

    # Extraer embeddings y descartar aquellos que contengan NaN
    embeddings_list = []
    valid_indices = []
    for idx, name in group['product_name'].iteritems():
        emb = get_embeddings(name)
        if np.isnan(emb).any():
            print(f"  - El embedding para '{name}' contiene NaN y se descarta.")
        else:
            embeddings_list.append(emb)
            valid_indices.append(idx)

    if not embeddings_list:
        print(f"  - No se obtuvieron embeddings válidos para {country}.")
        continue

    # Filtrar el grupo según los índices válidos
    group = group.loc[valid_indices]
    X_embeddings = np.array(embeddings_list)
    X = csr_matrix(X_embeddings)
    y = group['Category']

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo (RandomForest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  - Accuracy para {country}: {acc}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Guardar el modelo (no se utiliza vectorizador al usar embeddings)
    models_by_country[country] = {'model': model, 'vectorizer': None}

#############################################
# 3. Preprocesamiento y predicción en df_products
#############################################
# Precomputar embeddings para todos los nombres de productos en df_products si aún no existen
if 'precomputed_embeddings' not in df_products.columns:
    print("\nPrecomputando embeddings para df_products...")
    # Asegurarse de que product_name no tenga valores nulos
    df_products = df_products.dropna(subset=['product_name'])
    df_products['precomputed_embeddings'] = df_products['product_name'].apply(get_embeddings)
    print("Embeddings precomputados.")


# Función para predecir la categoría por país utilizando únicamente 'product_name'
def predict_for_country(country, group):
    if country not in models_by_country:
        print(f"Advertencia: No hay modelo entrenado para el país: {country}")
        return None
    print(f"Prediciendo categorías para el país: {country}")
    model = models_by_country[country]['model']
    # Asegurarse de que no haya valores nulos en product_name
    group = group.dropna(subset=['product_name']).copy()
    if group.empty:
        print(f"  - El grupo para {country} está vacío tras eliminar nulos.")
        return None
    # Asegurarse de que los embeddings sean válidos
    valid_embeddings = []
    valid_indices = []
    for idx, emb in group['precomputed_embeddings'].iteritems():
        if np.isnan(emb).any():
            print(f"  - Embedding inválido en índice {idx} para {country}.")
        else:
            valid_embeddings.append(emb)
            valid_indices.append(idx)
    if not valid_embeddings:
        print(f"  - No hay embeddings válidos para {country} en df_products.")
        return None
    group = group.loc[valid_indices]
    embeddings_batch = np.vstack(valid_embeddings)
    X_features = csr_matrix(embeddings_batch)
    predictions = model.predict(X_features)
    group['predicted_category'] = predictions
    return group


# Agrupar df_products por país y procesar en paralelo
grouped_products = df_products.groupby('order_country_code')
results = Parallel(n_jobs=-1)(
    delayed(predict_for_country)(country, group) for country, group in grouped_products
)

# Filtrar resultados no nulos y concatenar
results = [res for res in results if res is not None]
if results:
    df_products_predicted = pd.concat(results, ignore_index=True)
    # (Opcional) Eliminar la columna de embeddings precomputados para limpiar el DataFrame final
    if 'precomputed_embeddings' in df_products_predicted.columns:
        df_products_predicted = df_products_predicted.drop(columns=['precomputed_embeddings'])
    print("\nDataFrame con categorías predichas:")
    print(df_products_predicted.head())
else:
    print("No se obtuvieron predicciones para ningún país.")

df_products_predicted.to_csv("predicciones_productosKZ2.csv", index=False)
