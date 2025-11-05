import pandas as pd
import datetime
import trino
import matplotlib.pyplot as plt
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


# Lista de países a procesar
countries = ['AM']
all_orders = []

# -----------------------------
# Extraer órdenes por país
# -----------------------------
for country in countries:
    query_orders = f'''
WITH
  -- 1. CTE Base: Órdenes en los últimos 24 meses (columnas base y filtros esenciales)
  customer_orders_base AS (
    SELECT
      od.customer_id,
      od.order_country_code,
      od.order_id,
      od.order_activated_local_at,
      od.order_is_prime,
      od.order_vertical,
      od.order_final_status,
      od.order_parent_relationship_type,
      od.store_name,
      od.store_id,
      od.order_city_code,
      od.order_transacted_value_eur,
      od.order_picked_up_by_courier_at,
      od.order_courier_arrival_to_pickup_at,
      od.order_created_at,
      od.order_terminated_at,
      od.order_is_first_delivered_order,
      p.payment_method
    FROM delta.central_order_descriptors_odp.order_descriptors_v2 AS od
    LEFT JOIN delta.fintech_payments_odp.payments p ON od.order_id = p.order_id
    WHERE
      date_trunc('month', od.order_activated_local_at) >= date_trunc('month', date_add('month', -24, current_date))
      AND od.order_country_code IN ('AM')
      AND od.order_final_status = 'DeliveredStatus'
      AND od.order_parent_relationship_type IS NULL
  ),
  -- MODIFICACIÓN: Encontrar la primera orden entregada *primero* desde la base
  CustomerFirstDeliveredOrderBase AS (
      SELECT
          customer_id,
          order_country_code,
          order_city_code,
          order_activated_local_at AS customer_first_delivered_date_time,
          order_id,
          ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_activated_local_at ASC) as rn
      FROM customer_orders_base
      WHERE order_is_first_delivered_order = true -- Keep this filter as a first step
  ),
  -- MODIFICACIÓN: Unir solo las primeras órdenes a las tablas de eventos si es necesario
  CustomerFirstDeliveredOrderInfo AS (
      SELECT
          cfdb.customer_id,
          cfdb.order_country_code,
          cfdb.order_city_code,
          cfdb.customer_first_delivered_date_time,
          -- Join to event tables *only* for the first order to get H3
          h3_cell_to_parent(h3_string_to_h3(se.h8_hexagon),7) as h7_hexagon
      FROM CustomerFirstDeliveredOrderBase cfdb
      LEFT JOIN  delta.customer_behaviour_odp.enriched_backend_event__checkout_order_created_v3 od ON cfdb.order_id = od.order_id -- Join on the first order's ID
      LEFT JOIN delta.customer_behaviour_odp.dynamic_sessions_v1 se  ON od.dynamic_session_id = se.dynamic_session_id -- Join on the session ID from the first order event
      WHERE cfdb.rn = 1 -- Select only the first order per customer
  ),
  -- MODIFICACIÓN: Asegurar una sola fila por cliente en installs (using distinct for simplicity)
  installs AS (
      SELECT DISTINCT -- Use DISTINCT if GROUP BY is too complex, assuming (user_id, network, organic) is unique per user or first relevant is sufficient
        app_install_user_id as customer_id,
        app_install_network_name,
        app_install_is_organic
      FROM delta.growth_adjust_odp.adjust_install_v3 i
      WHERE app_install_user_id IS NOT NULL -- Add null check
      -- GROUP BY 1, 2, 3 -- Using ordinals based on SELECT list
  ),
  -- CTEs for joins (usan customer_orders_base) - No changes here unless needed based on review above
  pricing_discounts AS (
      -- Consider adding GROUP BY order_id if multiple rows per order are possible even with DISTINCT
      SELECT DISTINCT order_id, discount_type FROM delta.growth_pricing_discounts_odp.pricing_discounts WHERE discount_type='PROMOTOOL'
  ),
  pnl_order_level AS (
      SELECT order_id, contribution_margin_eur FROM delta.finance_financial_reports_odp.pnl_order_level
  ),
  monthly_stores AS (
      SELECT
        customer_id,
        date_trunc('month', order_activated_local_at) as month,
        COUNT(DISTINCT store_name) AS stores
      FROM customer_orders_base
      GROUP BY 1, 2 -- Uses ordinals based on SELECT list
  ),
  favorite_stores AS (
      SELECT
        order_country_code,
        customer_id,
        store_name,
        COUNT(DISTINCT order_id) AS orders
      FROM customer_orders_base
      GROUP BY 1, 2 , 3 -- Uses ordinals based on SELECT list
  ),
  store_segmentation AS(
       -- Consider adding GROUP BY store_id if multiple rows per store_id are possible even with DISTINCT
       SELECT DISTINCT
         store_id,
         segmentation
         FROM delta.partner_segmentation_odp.daily_partner_segmentation WHERE p_end_date is null and country_code IN ('AM')
    ),
  top_partners AS (
      SELECT
        order_country_code,
        store_id
      FROM (
        SELECT
          order_country_code,
          store_name,
          store_id,
          ROW_NUMBER() OVER (PARTITION BY order_country_code ORDER BY COUNT(DISTINCT order_id) DESC) AS rn -- 4
        FROM customer_orders_base
        WHERE store_name IS NOT NULL
        GROUP BY 1, 2, 3
      ) t
      WHERE rn <= 10
  ),
  top_cities AS (
      SELECT
        order_country_code,
        order_city_code
      FROM (
        SELECT
          order_country_code,
          order_city_code,
          ROW_NUMBER() OVER (PARTITION BY order_country_code ORDER BY COUNT(DISTINCT order_id) DESC) AS rn
        FROM customer_orders_base
        GROUP BY 1, 2 -- Uses ordinals based on SELECT list
      ) ranked_cities
      WHERE rn = 1
  ),
  -- CTE: Identifica la primera orden Qcommerce entregada por cliente para obtener la fecha
  CustomerFirstQcommOrderInfo AS (
      SELECT DISTINCT
          co.customer_id,
          fo.first_order_in_vertical_date AS customer_first_QComm_date_time
      FROM customer_orders_base co
      LEFT JOIN delta.mfc__first_order_levels__odp.first_order_levels fo ON fo.order_id=co.order_id
      WHERE store_vertical='QCommerce' -- Ensure this vertical filter is appropriate
  ),
  -- CTE: Identifica la primera orden Prime entregada por cliente para obtener la fecha
  CustomerFirstPrimeOrderInfo AS (
      SELECT
          co.customer_id,
          MIN(co.order_activated_local_at) AS customer_first_Prime_date_time
      FROM customer_orders_base co
      WHERE co.order_is_prime = true
      GROUP BY 1 -- Uses ordinal based on SELECT list
  ),
  -- 2. CTE: Prepara los datos a nivel de orden uniendo info de la primera orden entregada y joins adicionales
  orders_with_context AS (
    SELECT
      co.customer_id,
      cfdd.order_country_code,
      co.order_id,
      co.order_activated_local_at,
      co.order_is_prime,
      co.order_vertical,
      co.payment_method,
      pr.discount_type AS promo_type,
      s.stores AS num_monthly_stores,
      ss.segmentation, -- 10
      co.order_transacted_value_eur, -- 11
      CASE WHEN co.order_final_status = 'CanceledStatus' THEN 1 ELSE 0 END AS is_canceled,
      cm.contribution_margin_eur,
      -- Flags calculated here
      CASE WHEN tp.store_id IS NOT NULL THEN 1 ELSE 0 END AS is_top_partner,
      CASE WHEN rc.order_city_code IS NOT NULL THEN 1 ELSE 0 END AS is_top_city,
      co.order_city_code AS order_city_code_from_order,
      cfdd.order_city_code AS first_order_city_code,
      -- Información de la primera orden entregada unida
      cfdd.customer_first_delivered_date_time,
      cfdd.h7_hexagon,
      fq.customer_first_QComm_date_time,
      fp.customer_first_Prime_date_time,
      co.store_id
    FROM customer_orders_base co
    INNER  JOIN CustomerFirstDeliveredOrderInfo cfdd ON co.customer_id = cfdd.customer_id -- INNER JOIN here is fine, as customer_metrics groups by customer_id
    LEFT JOIN CustomerFirstQcommOrderInfo fq ON co.customer_id = fq.customer_id
    LEFT JOIN CustomerFirstPrimeOrderInfo fp ON co.customer_id = fp.customer_id
    LEFT JOIN pricing_discounts pr ON co.order_id = pr.order_id
    LEFT JOIN pnl_order_level cm ON co.order_id = cm.order_id
    -- Check logic here: s.stores is a monthly count, joining it per order seems wrong.
    -- This join might not be needed here if 'num_monthly_stores' is calculated in customer_metrics directly.
    LEFT JOIN monthly_stores s ON s.customer_id = co.customer_id AND date_trunc('month', s.month) = date_trunc('month', co.order_activated_local_at)
    LEFT JOIN top_partners tp ON co.order_country_code = tp.order_country_code AND co.store_id = tp.store_id
    LEFT JOIN top_cities rc ON co.order_country_code = rc.order_country_code AND co.order_city_code = rc.order_city_code
    LEFT JOIN store_segmentation ss ON ss.store_id=co.store_id
  ),
  -- 3. CTE: Agregación de métricas por cliente
  customer_metrics AS (
    SELECT
      oc.customer_id,
      -- Use country and city from the first delivered order for customer-level
      MAX(oc.order_country_code) AS order_country_code,
      MAX(oc.first_order_city_code) AS customer_city,
      MAX(oc.h7_hexagon) AS h7_hexagon,
      COUNT(DISTINCT oc.order_id) AS total_orders,

      -- Métricas de número de órdenes en ventanas de tiempo
      COUNT(DISTINCT CASE WHEN date_diff('day', date(oc.customer_first_delivered_date_time), date(oc.order_activated_local_at)) BETWEEN 0 AND 6 THEN oc.order_id  END) AS num_orders_days_1_to_7,
      COUNT(DISTINCT CASE WHEN date_diff('day', date(oc.customer_first_delivered_date_time), date(oc.order_activated_local_at)) BETWEEN 7 AND 13 THEN oc.order_id END) AS num_orders_days_8_to_14,
      COUNT(DISTINCT CASE WHEN date_diff('day', date(oc.customer_first_delivered_date_time), date(oc.order_activated_local_at)) BETWEEN 14 AND 20 THEN oc.order_id END) AS num_orders_days_15_to_21,
      COUNT(DISTINCT CASE WHEN date_diff('day', date(oc.customer_first_delivered_date_time), date(oc.order_activated_local_at)) BETWEEN 21 AND 27 THEN oc.order_id END) AS num_orders_days_22_to_28,
      COUNT(DISTINCT CASE WHEN date_diff('day', date(oc.customer_first_delivered_date_time), date(oc.order_activated_local_at)) BETWEEN 0 AND 29 THEN oc.order_id END) AS num_orders_days_1_to_30_total,
      COUNT(DISTINCT CASE WHEN date_diff('day', date(oc.customer_first_delivered_date_time), date(oc.order_activated_local_at)) BETWEEN 30 AND 59 THEN oc.order_id END) AS num_orders_days_31_to_60,
      COUNT(DISTINCT CASE WHEN date_diff('day', date(oc.customer_first_delivered_date_time), date(oc.order_activated_local_at)) BETWEEN 60 AND 89 THEN oc.order_id END) AS num_orders_days_61_to_90,
      COUNT(DISTINCT CASE WHEN date_diff('day', date(oc.customer_first_delivered_date_time), date(oc.order_activated_local_at)) BETWEEN 90 AND 119 THEN oc.order_id END) AS num_orders_days_91_to_120,
      COUNT(DISTINCT CASE WHEN date_diff('day', date(oc.customer_first_delivered_date_time), date(oc.order_activated_local_at)) BETWEEN 120 AND 149 THEN oc.order_id END) AS num_orders_days_121_to_150,
      COUNT(DISTINCT CASE WHEN date_diff('day', date(oc.customer_first_delivered_date_time), date(oc.order_activated_local_at)) BETWEEN 150 AND 179 THEN oc.order_id END) AS num_orders_days_151_to_180,
      COUNT(DISTINCT CASE WHEN date_diff('day', date(oc.customer_first_delivered_date_time), date(oc.order_activated_local_at)) BETWEEN 0 AND 364 THEN oc.order_id END) AS num_orders_days_1_to_365_total,

      -- Total counts by time/day/type/method/segmentation/cancellation - used for percentage calculation below

      AVG(oc.order_transacted_value_eur) AS avg_transacted_value_eur,
      AVG(oc.contribution_margin_eur) AS avg_contribution_margin_eur,

      -- Percentage metrics calculated here using total_orders from the same group
      CAST(SUM(CASE WHEN hour(oc.order_activated_local_at) BETWEEN 0 AND 6 THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_orders_night,
      CAST(SUM(CASE WHEN hour(oc.order_activated_local_at) BETWEEN 7 AND 12 THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_orders_morning,
      CAST(SUM(CASE WHEN hour(oc.order_activated_local_at) BETWEEN 13 AND 18 THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_orders_afternoon,
      CAST(SUM(CASE WHEN hour(oc.order_activated_local_at) BETWEEN 19 AND 23 THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_orders_evening,
      CAST(SUM(CASE WHEN day_of_week(oc.order_activated_local_at) BETWEEN 2 AND 6 THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_orders_weekday,
      CAST(SUM(CASE WHEN day_of_week(oc.order_activated_local_at) = 2 THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_orders_monday,
      CAST(SUM(CASE WHEN day_of_week(oc.order_activated_local_at) = 3 THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_orders_tuesday,
      CAST(SUM(CASE WHEN day_of_week(oc.order_activated_local_at) = 4 THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_orders_wednesday,
      CAST(SUM(CASE WHEN day_of_week(oc.order_activated_local_at) = 5 THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_orders_thursday,
      CAST(SUM(CASE WHEN day_of_week(oc.order_activated_local_at) = 6 THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_orders_friday,
      CAST(SUM(CASE WHEN day_of_week(oc.order_activated_local_at) = 7 THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_orders_saturday,
      CAST(SUM(CASE WHEN day_of_week(oc.order_activated_local_at) = 1 THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_orders_sunday,
      CAST(SUM(CASE WHEN oc.order_is_prime = true THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_prime_orders,
      CAST(SUM(CASE WHEN oc.order_vertical = 'QCommerce' THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_qcommerce_orders,
      CAST(SUM(CASE WHEN oc.promo_type = 'PROMOTOOL' THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_promo_orders,
      CAST(SUM(CASE WHEN oc.payment_method = 'CREDIT_CARD' THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_credit_card_orders,
      CAST(SUM(CASE WHEN oc.payment_method = 'CASH' THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_cash_orders,
      CAST(SUM(CASE WHEN oc.payment_method = 'APPLE_PAY' THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_apple_pay_orders,
      CAST(SUM(CASE WHEN oc.payment_method = 'GOOGLE_PAY' THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_google_pay_orders,
      CAST(SUM(CASE WHEN oc.payment_method = 'BANK_TRANSFER' THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_bank_transfer_orders,
      CAST(SUM(CASE WHEN oc.payment_method = 'KASPI' THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_kaspi_orders,
      CAST(SUM(CASE WHEN oc.payment_method = 'BLIK' THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_blik_orders,
      CAST(SUM(CASE WHEN oc.segmentation = 'Top City' THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_top_city_orders,
      CAST(SUM(CASE WHEN oc.segmentation = 'Top Country' THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_top_country_orders,
      CAST(SUM(CASE WHEN oc.segmentation = 'Local Selection' THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_local_selection_orders,
      CAST(SUM(CASE WHEN oc.segmentation = 'Long Tail' THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_long_tail_orders,
      CAST(SUM(CASE WHEN oc.segmentation = 'Big Chain' THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_big_chain_orders,
      CAST(SUM(CASE WHEN oc.is_top_partner = 1 THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(DISTINCT oc.order_id), 0) AS pct_top_partner_orders,


      -- Date metrics
      MAX(date(oc.customer_first_delivered_date_time)) AS customer_first_delivered_date,
      MAX(date(oc.customer_first_QComm_date_time)) AS customer_first_QComm_date_time,
      MAX(date(oc.customer_first_Prime_date_time)) AS customer_first_Prime_date_time,
      MAX(oc.order_activated_local_at) AS last_order_date,

      -- Number of distinct cities ordered from
      COUNT(DISTINCT oc.order_city_code_from_order) AS num_distinct_cities

    FROM orders_with_context oc
    GROUP BY
      1 -- oc.customer_id is the first column in the SELECT list
  ),

  -- 4. CTE: Cálculo del promedio de órdenes mensuales por usuario (sin cambios)
  monthly_orders AS (
    SELECT
      customer_id, -- 1
      AVG(CAST(COUNT_ORDERS AS DOUBLE)) AS avg_monthly_orders -- 2
    FROM (
      SELECT
        customer_id,
        date_trunc('month', order_activated_local_at) AS month,
        COUNT(DISTINCT order_id) AS COUNT_ORDERS
      FROM customer_orders_base
      GROUP BY 1, 2 -- Already uses ordinals
    ) AS subquery
    GROUP BY
      1 -- customer_id is the first column in the outer SELECT list
  ),
  -- 5. CTE: Clasificación de usuarios (user_type) (sin cambios)
  user_type AS (
    SELECT
      mo.customer_id, -- 1
      CASE
        WHEN mo.avg_monthly_orders > 6 THEN 'A'
        WHEN mo.avg_monthly_orders < 3 THEN 'C'
        ELSE 'B'
      END AS user_type -- 2
    FROM monthly_orders mo
  ),
   favorite_stores_count AS (
    SELECT
      fs.order_country_code, -- 1
      fs.customer_id, -- 2
      fs.store_name, -- 3
      SUM(fs.orders) as orders, -- 4
      ROW_NUMBER() OVER (
      PARTITION BY fs.order_country_code, fs.customer_id
      ORDER BY SUM(fs.orders) DESC
    ) AS store_rank -- 5
    FROM favorite_stores fs -- Use favorite_stores directly
    GROUP BY 1, 2, 3 -- Uses ordinals based on SELECT list
        ),
top_stores_pivot AS (
  SELECT
    order_country_code,
    customer_id,
    MAX(CASE WHEN store_rank = 1 THEN store_name END) AS top_1_store,
    MAX(CASE WHEN store_rank = 2 THEN store_name END) AS top_2_store,
    MAX(CASE WHEN store_rank = 3 THEN store_name END) AS top_3_store

  FROM favorite_stores_count
  WHERE store_rank <= 3
  GROUP BY 1, 2 -- Already uses ordinals
) ,
-- Final CTE to join all customer-level information
FinalCustomerProfile AS (
    SELECT
        cm.*, -- Select all metrics from customer_metrics
        ut.user_type, -- Add user type
        mo.avg_monthly_orders, -- Add average monthly orders
        -- Calculate date differences and avg since active here using dates from cm
        DATE_DIFF('day', cm.customer_first_delivered_date, current_date) AS days_since_first_order,
        DATE_DIFF('day', cm.customer_first_delivered_date, cm.customer_first_QComm_date_time) AS days_until_first_Qcomm,
        DATE_DIFF('day', cm.customer_first_delivered_date, cm.customer_first_Prime_date_time) AS days_until_first_Prime,
        CASE
            WHEN DATE_DIFF('day', cm.customer_first_delivered_date, current_date) = 0 THEN CAST(cm.total_orders AS DOUBLE)
            ELSE CAST(cm.total_orders AS DOUBLE) / (NULLIF(DATE_DIFF('day', cm.customer_first_delivered_date, current_date), 0) / 30.0) -- Use NULLIF here too
        END AS avg_monthly_orders_since_active,
         -- Install info
        i.app_install_is_organic as organic_installs,
        CASE WHEN i.app_install_network_name like 'Google Ads%' THEN true ELSE false END as google_installs,
        CASE WHEN i.app_install_network_name ='Apple Search Ads' THEN true ELSE false END as apple_installs,
        CASE WHEN i.app_install_network_name= 'Aura ironSource' THEN true ELSE false END as iron_source_installs,
        CASE WHEN i.app_install_network_name= 'Untrusted Devices'THEN true ELSE false END as untrusted_devices_installs,
        CASE WHEN i.app_install_network_name like 'Tiktok%' THEN true ELSE false END as tiktok_installs,
        -- Top stores info
        t3.top_1_store,
        t3.top_2_store,
        t3.top_3_store

    FROM customer_metrics cm
    JOIN user_type ut ON cm.customer_id = ut.customer_id
    JOIN monthly_orders mo ON cm.customer_id = mo.customer_id
    LEFT JOIN installs i ON cm.customer_id = i.customer_id
    LEFT JOIN top_stores_pivot t3 ON cm.customer_id = t3.customer_id AND cm.order_country_code = t3.order_country_code
  )

-- Final SELECT statement
SELECT *
FROM FinalCustomerProfile
'''
    with trino.dbapi.connect(**conn_details) as conn:
        df1 = pd.read_sql_query(query_orders, conn)
    all_orders.append(df1)

    # Ejecutar la consulta
    df = pd.read_sql(query_orders, conn)

    print(df.head())



# --- Data Cleaning and Preparation ---
if df is not None:
    print("\nPreparando datos...")
    # Inspeccionar tipos de datos y valores faltantes
    print(df.info())
    print(df.isnull().sum().sort_values(ascending=False)) # Mostrar columnas con más NaNs

    # Manejar valores faltantes
    # Identificar columnas numéricas y categóricas
    numerical_cols = df.select_dtypes(include=['number', 'bool']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Excluir columnas que no son features o que serán el target/ID
    exclude_cols = ['customer_id', 'user_type', 'customer_first_delivered_date',
                    'customer_first_QComm_date_time', 'customer_first_Prime_date_time',
                    'last_order_date', 'order_country_code', 'customer_city', 'h7_hexagon', # Excluir geografía cruda si no se va a usar directamente
                    'top_1_store', 'top_2_store', 'top_3_store' # Los nombres de tiendas son categóricos de alta cardinalidad, usaremos counts/flags
                   ] # Ajusta esta lista según qué columnas quieras usar como features

    feature_cols = [col for col in numerical_cols + categorical_cols if col not in exclude_cols]

    # Imputar valores faltantes
    # Para columnas numéricas: usar la mediana (más robusta a outliers)
    # Para columnas categóricas: usar la moda
    from sklearn.impute import SimpleImputer
    import numpy as np

    # Manejar posibles infinitos (ej: de divisiones por cero) reemplazándolos con NaN para imputación
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Imputar numéricas
    numerical_features = df[feature_cols].select_dtypes(include=['number']).columns
    if len(numerical_features) > 0:
        imputer_numerical = SimpleImputer(strategy='median')
        df[numerical_features] = imputer_numerical.fit_transform(df[numerical_features])

    # Imputar categóricas (convertir a string primero para asegurar)
    categorical_features = df[feature_cols].select_dtypes(include=['object', 'category']).columns
    if len(categorical_features) > 0:
         # Convertir a string para asegurar que SimpleImputer funcione correctamente con 'constant' o 'most_frequent'
        for col in categorical_features:
             df[col] = df[col].astype(str) # Asegura que NaN se convierta a 'nan' string o similar

        imputer_categorical = SimpleImputer(strategy='most_frequent') # O 'constant', fill_value='missing'
        df[categorical_features] = imputer_categorical.fit_transform(df[categorical_features])


    # --- Codificar variables categóricas para ML ---
    # One-Hot Encoding es común para modelos como Regresión Logística o SVM.
    # Los modelos de árbol (Random Forest, GBDT) a menudo manejan categóricas directamente o son menos sensibles a la codificación.
    # Para la correlación y algunos modelos, necesitamos codificar.
    df_encoded = pd.get_dummies(df, columns=categorical_features, dummy_na=False) # dummy_na=True si NaN imputado es una categoría válida

    # Asegurarse de que el target 'user_type' sea categórico para los modelos de clasificación
    df_encoded['user_type'] = df_encoded['user_type'].astype('category')
    # Define el orden si lo quieres tratar como ordinal (A > B > C)
    user_type_order = ['C', 'B', 'A']
    df_encoded['user_type'] = df_encoded['user_type'].cat.set_categories(user_type_order, ordered=True)


# Fase 2: Análisis de Correlación

# Para ver qué métricas se correlacionan con user_type, podemos usar:
# 1. Correlación de Spearman (si user_type se trata como ordinal)
# 2. Análisis visual (ej: box plots de métricas por tipo de usuario)
# 3. Pruebas estadísticas (ej: ANOVA para métricas numéricas, Chi-cuadrado para categóricas)

if df is not None:
    print("\nAnalizando correlación con User Type...")

    # Opción 1: Correlación de Spearman (User Type como ordinal)
    # Necesitamos una representación numérica de user_type
    user_type_map_corr = {'C': 1, 'B': 2, 'A': 3}
    df_encoded['user_type_ordinal'] = df_encoded['user_type'].map(user_type_map_corr)

    # Seleccionar solo columnas numéricas (incluyendo las one-hot encoded) y el target ordinal
    # Excluir el customer_id y el user_type original
    numerical_and_encoded_cols = df_encoded.select_dtypes(include=['number', 'bool']).columns
    correlation_cols = [col for col in numerical_and_encoded_cols if col != 'user_type_ordinal']
    correlation_df = df_encoded[correlation_cols + ['user_type_ordinal']]


    # Calcular correlación de Spearman
    # Spearman es adecuado porque user_type es ordinal y las distribuciones de tus métricas pueden no ser normales.
    try:
        correlation_matrix = correlation_df.corr(method='spearman')

        # Obtener las correlaciones con user_type_ordinal y ordenar
        user_type_correlations = correlation_matrix['user_type_ordinal'].sort_values(ascending=False)

        print("\nCorrelaciones (Spearman) con User Type (Ordinal A>B>C):")
        # Mostrar las top N correlaciones positivas y negativas
        print(user_type_correlations.head(15)) # Top 15 positivas
        print("...")
        print(user_type_correlations.tail(15)) # Top 15 negativas

        # Opcional: Visualizar la matriz de correlación completa (puede ser grande)
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # plt.figure(figsize=(15, 12))
        # sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm') # annot=True si quieres ver los valores (puede ser ilegible)
        # plt.title('Matriz de Correlación (Spearman) con User Type Ordinal')
        # plt.show()

    except Exception as e:
         print(f"Error al calcular o visualizar la matriz de correlación: {e}")
         print("Asegúrate de que hay suficientes datos numéricos después de la codificación y la imputación.")


    # Opción 2: Análisis visual de distribuciones (Ejemplo con boxplot)
    # Selecciona algunas métricas clave para visualizar
    # metrics_to_plot = ['total_orders', 'avg_monthly_orders', 'total_transacted_value_eur', 'num_orders_days_1_to_30_total', 'pct_prime_orders']
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    #
    # print("\nVisualizando distribución de métricas por User Type:")
    # for metric in metrics_to_plot:
    #     if metric in df_encoded.columns:
    #         plt.figure(figsize=(8, 5))
    #         sns.boxplot(x='user_type', y=metric, data=df_encoded, order=user_type_order)
    #         plt.title(f'Distribución de {metric} por User Type')
    #         plt.show()
    #     else:
    #          print(f"Métrica '{metric}' no encontrada en el DataFrame.")


# Fase 3: Modelo Predictivo y Feature Importance

# Como mencioné, predecir "new customer" con datos de clientes existentes no es el objetivo.
# Un objetivo más realista y útil con este dataset sería:
# A) Predecir el *tipo de usuario actual* (`user_type`). Esto te ayuda a entender qué métricas son más discriminatorias entre tipos.
# B) (Más avanzado) Predecir si un usuario "subirá de nivel" en un período futuro (ej: un usuario 'C' se volverá 'B' o 'A' en los próximos 3 meses). Esto requeriría una query diferente que identifique el estado del usuario al inicio del período y su estado al final.
# C) Predecir si un usuario será *activo* en un período futuro (predicción de retención). Similar a B, requiere datos futuros.

# Dado el dataset actual, el mejor uso inmediato es el caso A: predecir el tipo de usuario actual para entender la importancia de las features.

if df is not None:
    print("\nEntrenando modelo para predecir User Type (A, B, C)...")

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    # from sklearn.preprocessing import StandardScaler # Necesario si usas modelos sensibles a la escala

    # --- Preparación de datos para el modelo ---
    # Features (X) y Target (y)
    # Usamos el DataFrame codificado (df_encoded)
    X = df_encoded.drop(columns=['customer_id', 'user_type', 'user_type_ordinal'] + exclude_cols)
    y = df_encoded['user_type'] # Target es la variable categórica A, B, C

    # Eliminar columnas con muy poca varianza si hay (opcional pero útil a veces)
    # low_variance_cols = X.columns[X.var() < 0.01] # Umbral bajo, ajusta si es necesario
    # X = X.drop(columns=low_variance_cols)
    # print(f"Columnas eliminadas por baja varianza: {list(low_variance_cols)}")


    # Asegurar que no haya columnas con NaN después de imputación (debería estar resuelto)
    if X.isnull().sum().sum() > 0:
        print("ADVERTENCIA: Todavía hay NaNs en las features después de la imputación.")
        # Imputar de nuevo o revisar el proceso de imputación
        imputer_final = SimpleImputer(strategy='median')
        X = imputer_final.fit_transform(X)
        X = pd.DataFrame(X, columns=df_encoded.drop(columns=['customer_id', 'user_type', 'user_type_ordinal'] + exclude_cols).columns) # Recuperar nombres


    # Separar datos en conjuntos de entrenamiento y prueba
    # Usamos stratify=y porque user_type puede tener clases desbalanceadas
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y) # 25% para prueba


    # Opcional: Escalado de features (importante para modelos basados en distancia/gradiente)
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    # X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns) # Opcional: mantener nombres de columnas
    # X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


    # --- Entrenar el modelo ---
    # Random Forest es un buen modelo para empezar: maneja bien features numéricas y categóricas (una vez codificadas),
    # no es sensible a la escala y proporciona importancia de features.
    # class_weight='balanced' es útil si las clases A, B, C tienen tamaños muy diferentes.
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1) # n_jobs=-1 usa todos los cores disponibles

    print("Entrenando modelo Random Forest...")
    # Si escalaste, usa X_train_scaled y X_test_scaled
    model.fit(X_train, y_train)
    print("Entrenamiento completo.")

    # --- Evaluar el modelo ---
    y_pred = model.predict(X_test)

    print("\nEvaluación del Modelo (Prediciendo User Type actual):")
    print(f"Accuracy (Precisión general): {accuracy_score(y_test, y_pred)}")
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=user_type_order)) # Usa los nombres de las clases
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))


    # --- Extraer Importancia de Features ---
    # Esto te dirá qué métricas (o features codificadas) fueron más importantes para que el modelo distinga entre tipos de usuario
    if hasattr(model, 'feature_importances_'):
        feature_importances = pd.Series(model.feature_importances_, index=X.columns)

        print("\nImportancia de Features según Random Forest:")
        # Mostrar las features más importantes (Top N)
        print(feature_importances.sort_values(ascending=False).head(30)) # Ajusta el número N

        # Opcional: Visualizar importancia de features
        # plt.figure(figsize=(12, 10))
        # feature_importances.sort_values(ascending=False).head(30).plot(kind='barh')
        # plt.title('Top 30 Importancia de Features')
        # plt.gca().invert_yaxis() # Invertir el eje para que la más importante esté arriba
        # plt.show()
    else:
        print("El modelo seleccionado no tiene atributo feature_importances_.")


    # --- Interpretación de Combinaciones de Métricas ---
    # La importancia de features te da la relevancia *individual* (o la contribución promedio en el caso de Random Forest)
    # Las combinaciones de métricas son más difíciles de identificar directamente con un simple análisis de correlación.
    # Sin embargo, modelos como Random Forest o Gradient Boosting aprenden interacciones entre features automáticamente.
    # Si una combinación fuera extremadamente potente, verías que las features individuales que la componen aparecen con alta importancia.
    # Para un análisis más profundo de interacciones, podrías usar técnicas como SHAP (SHapley Additive exPlanations) o LIME.
    # Por ahora, la importancia de features es un excelente punto de partida para saber en qué métricas centrar tu análisis manual.
