import pandas as pd
import re
# CHANGE 1: Import TfidfVectorizer instead of CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer

nltk.download("stopwords")
nltk.download("wordnet")

# ===============================
# PREPROCESSING SETUP
# ===============================

# Stopwoorden NL + ENG (Initial set for the cleaning function)
stop_words = set(stopwords.words("english")) | set(stopwords.words("dutch"))

# Lemmatizer voor Engels
lemmatizer_en = WordNetLemmatizer()

# Stemmer voor Nederlands
stemmer_nl = SnowballStemmer("dutch")

# Load Data
df = pd.read_csv("Uitgebreide_VKM_dataset_zonder_weird_data.csv")

def detect_language(text):
    # Mini-detectie op basis van veelgebruikte woorden
    dutch_keywords = ["de", "het", "een", "en", "je", "jij", "wij", "zijn", "module", "leren", "opleiding"]
    english_keywords = ["the", "a", "an", "and", "is", "are", "course", "learn"]

    text_low = text.lower()
    nl_score = sum(1 for w in dutch_keywords if w in text_low)
    en_score = sum(1 for w in english_keywords if w in text_low)

    return "nl" if nl_score >= en_score else "en"

def clean_text_nlp(text):
    if not isinstance(text, str) or text.strip() == "" or text.lower() in ["ntb", "tbd", "nader te bepalen"]:
        return "ntb"
    
    # Lowercase
    text = text.lower()

    # Verwijder cijfers en speciale tekens
    text = re.sub(r"[^a-zA-Záéíóúàèçäëïöüñ\s]", " ", text)

    # Normaliseer spaties
    text = re.sub(r"\s+", " ", text).strip()

    # Taal bepalen
    lang = detect_language(text)

    # Stopwoorden
    words = [w for w in text.split() if w not in stop_words]

    # NL = stemmer  
    if lang == "nl":
        words = [stemmer_nl.stem(w) for w in words]

    # EN = lemmatizer  
    else:
        words = [lemmatizer_en.lemmatize(w) for w in words]

    return " ".join(words) if words else "ntb"


# ===============================
# APPLY CLEANING
# ===============================

print("Cleaning text data...")
df["shortdescription"] = df["shortdescription"].apply(clean_text_nlp)
df["description"] = df["description"].apply(clean_text_nlp)

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.lower()

df["combined_text"] = (
    df["name"].astype(str) + " " +
    df["shortdescription"].astype(str) + " " +
    df["module_tags"].astype(str) + " " +
    df["location"].astype(str)
)

# ===============================
# TF-IDF VECTORIZATION
# ===============================

# Re-defining stopwords list for the vectorizer (optional redundancy, but safe)
stopwords_nl = stopwords.words("dutch")
stopwords_en = stopwords.words("english")
combined_stopwords = list(set(stopwords_nl + stopwords_en))

print(f"Start Vectorization with TF-IDF...")

# CHANGE 2: Use TfidfVectorizer
# You can tune min_df (ignore terms that appear in too few docs) 
# and max_df (ignore terms that appear in too many docs)
vectorizer = TfidfVectorizer(
    stop_words=combined_stopwords,
    min_df=1  # Optional: ignore words that only appear in 1 document
)

vectorized = vectorizer.fit_transform(df["combined_text"])

print(f"Matrix shape: {vectorized.shape}")

# Calculate Similarity
similarities = cosine_similarity(vectorized)
similarity_df = pd.DataFrame(similarities, index=df["name"], columns=df["name"])
print("Similariteit matrix gemaakt.")

# ===============================
# RECOMMENDATION FUNCTION
# ===============================

def recommend(module_name, similarity_df):
    try:
        # Sort descending to find highest similarity
        recs = similarity_df[module_name].sort_values(ascending=False)[1:6]
        print(f"\nAanbevolen modules voor '{module_name}':")
        for name, score in recs.items():
            print(f"- {name} (Score: {score:.2f})")
    except KeyError:
        print("\nModule niet gevonden in dataset.")
        print("Beschikbare modules (eerste 5):")
        print(similarity_df.index.tolist()[:5])

# Test
recommend("datagedreven besluitvorming met ai", similarity_df)