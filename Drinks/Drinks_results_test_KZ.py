import pandas as pd
import datetime
import trino

# -----------------------------
# Leer predicciones
# -----------------------------
file_path = "/Users/pedro.lucioglovoapp.com/PycharmProjects/PLECA/Drinks/test products_KZ_5.csv"
df = pd.read_csv(file_path)
df['predicted_category'] = df['predicted_category'].str.strip().str.lower()

# -----------------------------
# Contar productos con "чикен" o "хотдог" que NO estén clasificados como "other" (antes de la actualización)
# -----------------------------
mask_sauce = df['product_name'].str.contains("соус", case=False, na=False)
mask_sour = df['product_name'].str.contains("кисло", case=False, na=False)
mask_fries = df['product_name'].str.contains("фри", case=False, na=False)
mask_bread = df['product_name'].str.contains("хлеб", case=False, na=False)
mask_peperoni1 = df['product_name'].str.contains("Пепперони", case=False, na=False)
mask_peperoni2 = df['product_name'].str.contains("peperoni", case=False, na=False)
mask_peperoni3 = df['product_name'].str.contains("Пепероні", case=False, na=False)
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

mask_non_drink_items = (
    df['product_name'].str.contains("abaidoner", case=False, na=False) |
    df['product_name'].str.contains("aitvister", case=False, na=False) |
    df['product_name'].str.contains("chickenp", case=False, na=False) |
    df['product_name'].str.contains("turkebab", case=False, na=False) |
    df['product_name'].str.contains("hotdog", case=False, na=False) |
    df['product_name'].str.contains("soussyrnyi", case=False, na=False) |
    df['product_name'].str.contains("bigburger", case=False, na=False) |
    df['product_name'].str.contains("popcorn", case=False, na=False) |
    df['product_name'].str.contains("aitvisterboks", case=False, na=False) |
    df['product_name'].str.contains("roll", case=False, na=False) |
    df['product_name'].str.contains("kostnyibulon", case=False, na=False) |
    df['product_name'].str.contains("aitvisterkombo", case=False, na=False) |
    df['product_name'].str.contains("taiskiichaiszhemchugom", case=False, na=False) |
    df['product_name'].str.contains("filadelfiialiuks", case=False, na=False))


# Combinar todas las máscaras (los ítems de comida, no bebidas)
mask_combined = (mask_sauce | mask_sour | mask_fries | mask_bread |
                 mask_peperoni1 | mask_peperoni2 | mask_peperoni3 | mask_carbonara1 | mask_carbonara2 |
                 mask_pampushki1 | mask_pampushki2 | mask_cherrypie | mask_utensilios |
                 mask_fries2 | mask_bread2 | mask_sandwich | mask_chicken | mask_cream |
                 mask_non_drink_items)

count_non_other = df[mask_combined & (df['predicted_category'] != "other")].shape[0]
print("Productos con 'чикен' o 'хотдог' y categoría distinta a 'other':", count_non_other)

# -----------------------------
# Actualizar predicted_category a "other" para productos que incluyan "чикен" o "хотдог"
# -----------------------------
df.loc[mask_combined, 'predicted_category'] = "other"

# -----------------------------
# Actualizar productos relacionados con bebidas
# - Cambiar el nombre a "cocacola" para productos que contengan "coca" (o "кока")
# - Categorizar como "drink" productos que contengan "coca", "pepsi", "sprite", "fanta" o "agua" (o sus equivalentes en ruso)
# -----------------------------
mask_coca = (df['product_name'].str.contains("coca", case=False, na=False) |
             df['product_name'].str.contains("кока", case=False, na=False) |
             df['product_name'].str.contains("Koka", case=False, na=False) |
             df['product_name'].str.contains("Кола", case=False, na=False) |
             df['product_name'].str.contains("Kola", case=False, na=False))

mask_pepsi = (df['product_name'].str.contains("pepsi", case=False, na=False) |
              df['product_name'].str.contains("пепси", case=False, na=False))

mask_sprite = (df['product_name'].str.contains("sprite", case=False, na=False) |
               df['product_name'].str.contains("спрайт", case=False, na=False))

mask_fanta = (df['product_name'].str.contains("fanta", case=False, na=False) |
              df['product_name'].str.contains("фанта", case=False, na=False))

mask_agua = (df['product_name'].str.contains("water", case=False, na=False) |
             df['product_name'].str.contains("вода", case=False, na=False))

mask_kompot = (df['product_name'].str.contains("kompot", case=False, na=False) |
               df['product_name'].str.contains("компот", case=False, na=False))

# Más máscaras adicionales para bebidas populares
mask_mirinda = (df['product_name'].str.contains("mirinda", case=False, na=False) |
                df['product_name'].str.contains("миринда", case=False, na=False))

mask_redbull = (df['product_name'].str.contains("redbull", case=False, na=False) |
                df['product_name'].str.contains("редбул", case=False, na=False))

mask_lemonade = (df['product_name'].str.contains("lemonade", case=False, na=False) |
                 df['product_name'].str.contains("лимонад", case=False, na=False))

mask_7up = (df['product_name'].str.contains("7up", case=False, na=False))

mask_drinks_popular = (mask_coca | mask_pepsi | mask_sprite | mask_fanta |
                       mask_agua | mask_kompot | mask_mirinda | mask_redbull |
                       mask_lemonade | mask_7up)

# Cambiar el nombre a "cocacola" para productos que contengan "coca" o "кока"
df.loc[mask_coca, 'product_name'] = "cocacola"
# Categorizar todos los productos detectados como "drink"
df.loc[mask_drinks_popular, 'predicted_category'] = "drink"

# -----------------------------
# Guardar el DataFrame completo en un nuevo CSV
# -----------------------------
df.to_csv("test products_KZ_6.csv", index=False)
