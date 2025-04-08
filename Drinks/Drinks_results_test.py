import pandas as pd
import datetime
import trino

# -----------------------------
# Leer predicciones
# -----------------------------
file_path = "/Users/pedro.lucioglovoapp.com/PycharmProjects/PLECA/Drinks/test products_KZ_4.csv"
df = pd.read_csv(file_path)
df['predicted_category'] = df['predicted_category'].str.strip().str.lower()

# -----------------------------
# Contar productos con "чикен" o "хотдог" que NO estén clasificados como "other" (antes de la actualización)
# -----------------------------
mask_chicken = df['product_name'].str.contains("чикен", case=False, na=False)
mask_hotdog = df['product_name'].str.contains("хотдог", case=False, na=False)
mask_kebab = df['product_name'].str.contains("кебаб", case=False, na=False)
mask_burger = df['product_name'].str.contains("бургер", case=False, na=False)
mask_popcorn = df['product_name'].str.contains("попкорн", case=False, na=False)
mask_roll = df['product_name'].str.contains("ролл", case=False, na=False)


mask_combined = mask_chicken | mask_hotdog | mask_kebab | mask_burger | mask_popcorn | mask_roll

count_non_other = df[mask_combined & (df['predicted_category'] != "other")].shape[0]
print("Productos con 'чикен' o 'хотдог' y categoría distinta a 'other':", count_non_other)

# -----------------------------
# Actualizar predicted_category a "other" para productos que incluyan "чикен" o "хотдог"
# -----------------------------
df.loc[mask_combined, 'predicted_category'] = "other"

# -----------------------------
# Guardar el DataFrame completo en un nuevo CSV
# -----------------------------
df.to_csv("test products_KZ_5.csv", index=False)
