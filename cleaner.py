import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
# Laad de dataset
df = pd.read_csv("Uitgebreide_VKM_dataset.csv")

# ==========================================
# CONFIGURATIE
# ==========================================

empty_values = ["", "nan", "none", "null"]

weird_values = [
    "nvt", "volgt", "ntb", "nader te bepalen", "nog niet bekend",
    "nadert te bepalen", "nog te formuleren", "tbd", "n.n.b.", "navragen"
]

# Kolommen die we NIET als tekst willen behandelen (optioneel, voor veiligheid)
numeric_cols = ["id", "studycredit", "available_spots", "interests_match_score", "popularity_score"]

# Setup Stopwords voor Tags (Dutch + English)
stop_words = set(stopwords.words('english')) | set(stopwords.words('dutch'))
# Voeg extra ruiswoorden toe
extra_noise = {"ntb", "nan", "null", "none", "'", "['", "']"} 
stop_words.update(extra_noise)

# ==========================================
# HULPFUNCTIES
# ==========================================

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

def analyze_dataframe(df):
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

print("\nAnalyse voor opschoning:")
print(analyze_dataframe(df))
# ==========================================
# STAP 1: BASIS OPSCHONING
# ==========================================

# Verwijder de kleurenkolommen direct
cols_to_drop = ["Rood", "Groen", "Blauw", "Geel"]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Zet alles naar string, behalve als het puur numeriek moet blijven? 
# Voor TF-IDF is alles naar string prima.
df = df.astype(str)

# Lowercase en strip whitespace
df = df.apply(lambda col: col.str.lower().str.strip())

# Zet letterlijke strings "nan", "none" om naar "ntb"
for val in empty_values:
    df.replace(val, "ntb", inplace=True)

# ==========================================
# STAP 2: VEILIGE REPLACEMENT (FIX)
# ==========================================

# We gebruiken regex boundaries \b zodat "volgt" alleen wordt vervangen als het een los woord is,
# en niet als onderdeel van een ander woord (hoewel 'volgt' in een zin nog steeds riskant is).
# BETER: We vervangen alleen als de CEL bijna volledig uit deze term bestaat.

# Regex: ^ = begin cel, \s* = spaties, (lijst), \s* = spaties, $ = einde cel
# Dit vervangt dus alleen "  nog niet bekend  ", maar laat "De les volgt later" met rust.
safe_pattern = r'^\s*(' + '|'.join([re.escape(v) for v in weird_values]) + r')\s*$'

for col in df.columns:
    # Vervang exacte matches van weird values door 'ntb'
    df[col] = df[col].replace(to_replace=safe_pattern, value="ntb", regex=True)

# Specific Tag Cleaning (Stopword Removal) ---
def clean_tags_column(tag_string):
    if tag_string == 'ntb': return 'ntb'
    
    # 1. Remove list characters like [ ] ' "
    clean_str = re.sub(r"[\[\]'\"]", "", tag_string)
    
    # 2. Split by comma
    tags = clean_str.split(',')
    
    valid_tags = []
    for tag in tags:
        tag = tag.strip().lower()
        # 3. Filter: Must not be a stopword, must be > 1 char, must not be numeric
        if tag and tag not in stop_words and len(tag) > 1 and not tag.isdigit():
            valid_tags.append(tag)
            
    # Return as a clean comma-separated string (easier for reading) 
    # or keep as list string if preferred. Here we join them.
    return ", ".join(valid_tags) if valid_tags else "ntb"

print("Removing stopwords from module_tags...")
df['module_tags'] = df['module_tags'].apply(clean_tags_column)

# ==========================================
# STAP 3: INTELLIGENT VULLEN (FIX)
# ==========================================

def fill_short_smart(row):
    short = row.get("shortdescription", "ntb")
    
    # Als shortdescription goed is, niks doen
    if short != "ntb" and short != "":
        return short

    # Haal backup velden op
    desc = row.get("description", "ntb")
    content = row.get("content", "ntb")
    
    valid_desc = desc != "ntb" and desc != ""
    valid_content = content != "ntb" and content != ""

    # Scenario 1: Beide zijn bruikbaar
    if valid_desc and valid_content:
        # CHECK: Als ze exact hetzelfde zijn (komt vaak voor in jouw dataset), pak er maar 1
        if desc == content:
            return desc
        # Anders samenvoegen
        return f"{desc} {content}"

    # Scenario 2: Alleen description
    if valid_desc:
        return desc

    # Scenario 3: Alleen content
    if valid_content:
        return content

    return "ntb"

print("Bezig met 'shortdescription' herstellen...")
df["shortdescription"] = df.apply(fill_short_smart, axis=1)



# ==========================================
# STAP 4: ANALYSE & OPSLAAN
# ==========================================

print("\nAnalyse na opschoning:")
print(analyze_dataframe(df))

output_file = "Uitgebreide_VKM_dataset_zonder_weird_data.csv"
df.to_csv(output_file, index=False)
print(f"\nKlaar! Opgeslagen als '{output_file}'")