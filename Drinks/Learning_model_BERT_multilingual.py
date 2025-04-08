import trino
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
import datetime
import re
from unidecode import unidecode

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

base_date = (datetime.date.today() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
countries = ['PL']
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
        AND order_parent_relationship_type IS NULL
        AND store_name IS NOT NULL
    GROUP BY store_name
    ORDER BY orders DESC
    LIMIT 300
), stores AS (
        SELECT DISTINCT
            order_country_code,
            store_address_id,
            store_name
        FROM delta.central_order_descriptors_odp.order_descriptors_v2 od
        WHERE od.order_final_status = 'DeliveredStatus'
          AND od.order_vertical = 'Food'
          AND od.store_name = 'McDonald''s'
          AND od.order_country_code = '{country}'
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
        ON s.store_address_id = pa.store_address_id
    INNER JOIN top_partners p ON s.store_name = p.store_name    
    WHERE CAST(pa.p_snapshot_date AS DATE) IN (SELECT day FROM max_day_cte)
      AND pa.product_is_available = TRUE
      AND pa.product_name IS NOT NULL
    '''
    with trino.dbapi.connect(**conn_details) as conn:
        df_tmp = pd.read_sql_query(query_products, conn)
    all_data.append(df_tmp)

df_products = pd.concat(all_data, ignore_index=True)
print("Datos extraídos de Starburst:")
print(df_products.head())

#############################################
# 2. Carga del dataset de entrenamiento y creación de modelos por país
#############################################
df_train = pd.read_csv("/Users/pedro.lucioglovoapp.com/PycharmProjects/PLECA/Drinks/PLECA orders with Drinks - Drinks Total.csv")

# Cargar modelo y tokenizer de Multilingual BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Función optimizada para obtener embeddings en batch
def get_bert_embeddings_batch(text_list, batch_size=16):
    embeddings = []
    device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")  # Usa GPU si está disponible
    bert_model.to(device)

    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = bert_model(**inputs)

        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Token [CLS]
        embeddings.extend(batch_embeddings)

    return np.array(embeddings)

# Entrenar modelos por país
models_by_country = {}
for country, group in df_train.groupby('order_country_code'):
    print(f"\nEntrenando modelo para {country}")
    group = group.dropna(subset=['product_name'])
    if group.empty:
        print(f"  - No hay datos en el grupo para {country}.")
        continue

    X_embeddings = get_bert_embeddings_batch(group['product_name'].tolist())
    X = csr_matrix(X_embeddings)
    y = group['Category']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  - Accuracy para {country}: {acc}")
    print(classification_report(y_test, y_pred, zero_division=0))

    models_by_country[country] = model

#############################################
# 3. Preprocesamiento y predicción en df_products
#############################################
if 'precomputed_embeddings' not in df_products.columns:
    print("\nPrecomputando embeddings en batch para df_products...")
    df_products = df_products.dropna(subset=['product_name'])
    df_products['precomputed_embeddings'] = list(get_bert_embeddings_batch(df_products['product_name'].tolist()))
    print("Embeddings precomputados.")

def predict_for_country(country, group):
    if country not in models_by_country:
        print(f"Advertencia: No hay modelo entrenado para {country}")
        return None

    print(f"Prediciendo categorías para {country}")
    model = models_by_country[country]
    group = group.dropna(subset=['product_name'])
    if group.empty:
        print(f"  - El grupo para {country} está vacío tras eliminar nulos.")
        return None

    # Convertir las embeddings precomputadas a una matriz CSR
    X_features = csr_matrix(np.vstack(group['precomputed_embeddings']))
    # Convertir los índices e indptr a np.int32 para evitar el error
    X_features.indices = X_features.indices.astype(np.int32)
    X_features.indptr = X_features.indptr.astype(np.int32)

    group['predicted_category'] = model.predict(X_features)
    return group

grouped_products = df_products.groupby('order_country_code')
results = Parallel(n_jobs=-1)(
    delayed(predict_for_country)(country, group) for country, group in grouped_products
)
results = [res for res in results if res is not None]
if results:
    df_products_predicted = pd.concat(results, ignore_index=True)
    df_products_predicted.drop(columns=['precomputed_embeddings'], inplace=True, errors='ignore')
    print("\nDataFrame con categorías predichas:")
    print(df_products_predicted.head())
    df_products_predicted.to_csv("McDonalds.csv", index=False)
else:
    print("No se obtuvieron predicciones.")
