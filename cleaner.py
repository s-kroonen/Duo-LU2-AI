import pandas as pd
import numpy as np
import re
df = pd.read_csv("Uitgebreide_VKM_dataset.csv")
df.head()

# Hulpfuncties en variabelen

empty_values = [
    "",           # lege strings
    "nan",        # NaN nadat alles string is
    "none",
    "null"
]

weird_values = [
    "nvt",
    "volgt", 
    "ntb",
    "nader te bepalen",
    "nog niet bekend",
    "nadert te bepalen",
    "nog te formuleren",
    "tbd",
    "n.n.b.",
    "nog niet bekend",
    "navragen"
]

def is_empty(value):
    """Checkt of waarde leeg, null of whitespace is."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False

def is_weird(value):
    """Checkt of waarde behoort tot weird values."""
    if not isinstance(value, str):
        return False
    val = value.lower().strip()
    return any(w in val for w in weird_values)

def is_ntb(value):
    """Checkt of waarde 'ntb' is."""
    return isinstance(value, str) and value.strip().lower() == "ntb"

def analyzedataframe(df):
    analysis = []

    for col in df.columns:
        total = len(df[col])

        empty_count = df[col].apply(is_empty).sum()
        weird_count = df[col].apply(is_weird).sum()
        ntb_count = df[col].apply(is_ntb).sum()

        general_error_count = empty_count + weird_count
        general_error_percent = round((general_error_count / total) * 100, 2)

        analysis.append({
            "column": col,

            # Empty
            "empty_values": empty_count,
            "empty_%": round((empty_count / total) * 100, 2),

            # Weird
            "weird_values": weird_count,
            "weird_%": round((weird_count / total) * 100, 2),

            # NTB after cleaning
            "ntb_after_cleaning": ntb_count,
            "ntb_%": round((ntb_count / total) * 100, 2),
            
            # Combined (sorting key)
            "general_error_total": general_error_count,
            "general_error_%": general_error_percent,
        })

    # Sorteren op hoogste foutpercentage
    analysis_df = pd.DataFrame(analysis)
    analysis_df = analysis_df.sort_values(by="general_error_%", ascending=False)
    return analysis_df


# Stap 0 — korte analyse van lege velden en "ntb"

print(analyzedataframe(df))


# Stap 1 — alles naar string voor consistente verwerking
df = df.astype(str)

# Stap 2 — passing op alle kolommen: strip en lowercase
# Lowercase-versie voor matching
df_lower = df.apply(lambda col: col.str.lower().str.strip())
df = df.apply(lambda col: col.str.lower().str.strip())
# Verwijder de onzinnige kleurkolommen
df = df.drop(columns=["Rood","Groen","Blauw","Geel"])

# Stap 3 — vervang ALLES wat in weird_values zit door "ntb"
pattern = "|".join([re.escape(v) for v in weird_values])

# masker dat True is als een waarde in een van de kolommen weird is
mask = df.apply(lambda col: col.astype(str).str.contains(pattern, case=False, na=False)).any(axis=1)

for col in df.columns:
    df[col] = df[col].astype(str).str.replace(pattern, "ntb", regex=True)


# Stap 4 — extra: lege velden of whitespace → NTB
# df = df.apply(lambda col: col.str.strip().replace("", "ntb"))
for value in empty_values:
    df[df_lower == value] = "ntb"

# Stap 5 Schoonmaken en vullen van lege kolommen
def fill_short(row):
    short = str(row.get("shortdescription", "")).strip().lower()

    # Alleen aanvullen als shortdescription == "ntb"
    if short != "ntb":
        return row["shortdescription"]

    # Description en content ophalen
    desc = str(row.get("description", "")).strip()
    content = str(row.get("content", "")).strip()

    # Check of ze bruikbaar zijn (niet "ntb" en niet leeg/None)
    valid_desc = desc.lower() != "ntb"
    valid_content = content.lower() != "ntb" 

    # Case 1 — beide bruikbaar → combineer
    if valid_desc and valid_content:
        return f"{desc} {content}".strip()

    # Case 2 — alleen description bruikbaar
    if valid_desc:
        return desc

    # Case 3 — alleen content bruikbaar
    if valid_content:
        return content

    # Case 4 — beide nutteloos → short blijft ntb
    return "ntb"


df["shortdescription"] = df.apply(fill_short, axis=1)


# Stap 6 — korte analyse van lege velden en "ntb" na opschoning
print(analyzedataframe(df))



df.to_csv("Uitgebreide_VKM_dataset_zonder_weird_data.csv", index=False)
print("Opschoning voltooid. Opgeslagen als 'Uitgebreide_VKM_dataset_zonder_weird_data.csv'")
