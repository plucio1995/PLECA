import datetime
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import trino  # Asegúrate de tener instalado el conector trino

# -------------------------
# Parámetros generales
# -------------------------
TARGET_COUNTRY = 'PL'  # País de interés
TRAIN_CSV = "PLECA orders with Drinks - Drinks Total.csv"  # Dataset de entrenamiento con datos de todos los países
OUTPUT_PREDICTIONS_CSV = "predicciones_productos_PL.csv"
MODEL_SAVE_DIR = "finetuned_multilingual_bert_drink_other"

# -------------------------
# 1. Extracción de productos disponibles desde Starburst
# -------------------------
HOST = 'starburst.g8s-data-platform-prod.glovoint.com'
PORT = 443
conn_details = {
    'host': HOST,
    'port': PORT,
    'http_scheme': 'https',
    'auth': trino.auth.OAuth2Authentication()
}

# Fecha base para la consulta (últimos 30 días)
base_date = (datetime.date.today() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')

query_products = f'''
WITH stores AS (
    SELECT DISTINCT
        order_country_code,
        store_address_id,
        store_name
    FROM delta.central_order_descriptors_odp.order_descriptors_v2 od
    WHERE od.order_final_status = 'DeliveredStatus'
      AND od.order_vertical = 'Food'
      AND od.order_country_code = '{TARGET_COUNTRY}'
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
WHERE CAST(pa.p_snapshot_date AS DATE) IN (SELECT day FROM max_day_cte)
  AND pa.product_is_available = TRUE
  AND pa.product_name IS NOT NULL
'''

print("Extrayendo productos disponibles desde Starburst...")
with trino.dbapi.connect(**conn_details) as conn:
    df_products_starburst = pd.read_sql_query(query_products, conn)
print("Datos extraídos:")
print(df_products_starburst.head())

# -------------------------
# 2. Preparación del dataset de entrenamiento y filtrado por país
# -------------------------
print("\nCargando dataset de entrenamiento...")
df_train = pd.read_csv(TRAIN_CSV)

# Filtrar para el país de interés (si existe la columna 'order_country_code')
if 'order_country_code' in df_train.columns:
    df_train = df_train[df_train['order_country_code'] == TARGET_COUNTRY]

# Asegurarse de tener datos y eliminar nulos en 'product_name'
df_train = df_train.dropna(subset=['product_name'])
print(f"Ejemplos de entrenamiento para {TARGET_COUNTRY}: {df_train.shape[0]} registros")

# Mapear las categorías a valores numéricos: 'drink' -> 1, 'other' -> 0
df_train['label'] = df_train['Category'].apply(lambda x: 1 if x.strip().lower() == 'drink' else 0)

# División en entrenamiento y validación
train_df, val_df = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train['label'])

# -------------------------
# 3. Definición de Dataset y DataLoader para entrenamiento y predicción
# -------------------------
class ProductDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = ProductDataset(
        texts=df.product_name.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

# Dataset para predicción (sin etiquetas)
class ProductDatasetForPrediction(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def create_data_loader_for_prediction(df, tokenizer, max_len, batch_size):
    ds = ProductDatasetForPrediction(
        texts=df.product_name.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)

# -------------------------
# 4. Fine Tuning de Multilingual BERT para clasificación binaria (drink vs other)
# -------------------------
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Parámetros de entrenamiento
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3

train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(val_df, tokenizer, MAX_LEN, BATCH_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    losses = []
    correct_predictions = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

print("\nIniciando Fine Tuning...")
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    train_acc, train_loss = train_epoch(model, train_data_loader, optimizer, device, scheduler)
    print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f}')

    val_acc, val_loss = eval_model(model, val_data_loader, device)
    print(f'Val   loss {val_loss:.4f} accuracy {val_acc:.4f}')
    print('-' * 30)

# Evaluación final en validación (opcional)
all_preds = []
all_labels = []
model.eval()
with torch.no_grad():
    for batch in val_data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
print("\nClassification Report en validación:")
print(classification_report(all_labels, all_preds, target_names=['other', 'drink']))

# Guardamos el modelo ajustado
model.save_pretrained(MODEL_SAVE_DIR)
tokenizer.save_pretrained(MODEL_SAVE_DIR)
print(f"Modelo guardado en {MODEL_SAVE_DIR}")

# -------------------------
# 5. Predicción sobre los productos extraídos de Starburst y guardado en CSV
# -------------------------
print("\nRealizando predicciones sobre productos extraídos desde Starburst...")
# Usamos df_products_starburst obtenido en el paso 1
df_products_pred = df_products_starburst.dropna(subset=['product_name'])

prediction_data_loader = create_data_loader_for_prediction(df_products_pred, tokenizer, MAX_LEN, BATCH_SIZE)

model.eval()
predictions = []
with torch.no_grad():
    for batch in prediction_data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
        predictions.extend(preds.cpu().numpy())

# Asumimos que 1 corresponde a "drink" y 0 a "other"
df_products_pred['predicted_category'] = ['drink' if pred == 1 else 'other' for pred in predictions]

# Guardamos las predicciones en un CSV
df_products_pred.to_csv(OUTPUT_PREDICTIONS_CSV, index=False)
print(f"Predicciones guardadas en {OUTPUT_PREDICTIONS_CSV}")
