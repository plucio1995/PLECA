import pandas as pd
import datetime
import trino

# -----------------------------
# Leer predicciones
# -----------------------------
file_path = "/Users/pedro.lucioglovoapp.com/PycharmProjects/PLECA/Drinks/test products_PL_6.csv.csv"
df = pd.read_csv(file_path)
df['predicted_category'] = df['predicted_category'].str.strip().str.lower()

# -----------------------------
# Actualizar productos relacionados con bebidas
# - Cambiar el nombre a "cocacola" para productos que contengan "coca" (o "кока")
# - Categorizar como "drink" productos que contengan "coca", "pepsi", "sprite", "fanta" o "agua" (o sus equivalentes en ruso)
# -----------------------------
mask_coca = (df['product_name'].str.contains("coca", case=False, na=False) |
             df['product_name'].str.contains("cola", case=False, na=False))

mask_pepsi = (df['product_name'].str.contains("pepsi", case=False, na=False) |
              df['product_name'].str.contains("пепси", case=False, na=False))

mask_sprite = (df['product_name'].str.contains("sprite", case=False, na=False) |
               df['product_name'].str.contains("спрайт", case=False, na=False))

mask_fanta = (df['product_name'].str.contains("fanta", case=False, na=False) |
              df['product_name'].str.contains("фанта", case=False, na=False))

mask_agua = (df['product_name'].str.contains("water", case=False, na=False) |
             df['product_name'].str.contains("вода", case=False, na=False) |
             df['product_name'].str.contains("woda", case=False, na=False))

mask_kompot = (df['product_name'].str.contains("kompot", case=False, na=False) |
               df['product_name'].str.contains("компот", case=False, na=False))

mask_jugo = (df['product_name'].str.contains("sok", case=False, na=False) |
             df['product_name'].str.contains("juice", case=False, na=False) |
             df['product_name'].str.contains("сок", case=False, na=False))

mask_tea = (df['product_name'].str.contains("herbata", case=False, na=False) |
            df['product_name'].str.contains("ice tea", case=False, na=False) |
            df['product_name'].str.contains("чай", case=False, na=False))

mask_kefir = (df['product_name'].str.contains("kefir", case=False, na=False) |
              df['product_name'].str.contains("кефир", case=False, na=False))

mask_zbyszko = df['product_name'].str.contains("zbyszko", case=False, na=False)

mask_oranzada = (df['product_name'].str.contains("oranżada", case=False, na=False) |
                 df['product_name'].str.contains("оранжада", case=False, na=False))

mask_lemonade = (df['product_name'].str.contains("Lemoniada", case=False, na=False))

mask_drink = (mask_coca | mask_pepsi | mask_sprite | mask_fanta | mask_agua |
              mask_kompot | mask_jugo | mask_tea | mask_kefir | mask_zbyszko | mask_oranzada | mask_lemonade)

# Cambiar el nombre a "cocacola"
df.loc[mask_coca, 'product_name'] = "cocacola"
# Categorizar como "drink"
df.loc[mask_drink, 'predicted_category'] = "drink"

# -----------------------------
# Contar productos con "чикен" o "хотдог" que NO estén clasificados como "other" (antes de la actualización)
# -----------------------------
mask_galleta = df['product_name'].str.contains("ciacho", case=False, na=False)
mask_postre = df['product_name'].str.contains("biala czekolada", case=False, na=False)
mask_helado = df['product_name'].str.contains("lody", case=False, na=False)
mask_bread = df['product_name'].str.contains("chikker", case=False, na=False)
mask_peperoni1 = df['product_name'].str.contains("McPops", case=False, na=False)
mask_peperoni2 = df['product_name'].str.contains("lody", case=False, na=False)
mask_peperoni3 = df['product_name'].str.contains("smaku waniliowym z polewą ", case=False, na=False)
mask_carbonara1 = df['product_name'].str.contains("Карбонара", case=False, na=False)
mask_carbonara2 = df['product_name'].str.contains("karbonara", case=False, na=False)
mask_pampushki1 = df['product_name'].str.contains("пампушки", case=False, na=False)
mask_pampushki2 = df['product_name'].str.contains("Pampushk", case=False, na=False)
mask_cherrypie = df['product_name'].str.contains("макпиріг", case=False, na=False)
mask_utensilios = df['product_name'].str.contains("прилади", case=False, na=False)
mask_fries2 = df['product_name'].str.contains("картопля", case=False, na=False)
mask_bread2 = df['product_name'].str.contains("хліб", case=False, na=False)
mask_sandwich = df['product_name'].str.contains("твістер", case=False, na=False)
mask_chicken = df['product_name'].str.contains("курячий", case=False, na=False)
mask_cream = df['product_name'].str.contains("сметана", case=False, na=False)

mask_combined = (mask_galleta | mask_postre | mask_helado | mask_bread |
                 mask_peperoni1 | mask_peperoni2 | mask_peperoni3 | mask_carbonara1 | mask_carbonara2 |
                 mask_pampushki1 | mask_pampushki2 | mask_cherrypie | mask_utensilios |
                 mask_fries2 | mask_bread2 | mask_sandwich | mask_chicken | mask_cream)

count_non_other = df[mask_combined & (df['predicted_category'] != "other")].shape[0]
print("Productos con 'чикен' o 'хотдог' y categoría distinta a 'other':", count_non_other)

# -----------------------------
# Actualizar predicted_category a "other" para productos que incluyan los patrones anteriores
# -----------------------------
df.loc[mask_combined, 'predicted_category'] = "other"

# -----------------------------
# Guardar el DataFrame completo en un nuevo CSV
# -----------------------------
df.to_csv("test products_PL_6.csv", index=False)
