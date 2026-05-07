import os
import re
import math
import string
import pickle

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ============================================================
# Configuration de la page  (doit être le 1er appel Streamlit)
# ============================================================
st.set_page_config(
    page_title="Analyse de Sentiments",
    page_icon="🛒",
    layout="centered",
)

# ============================================================
# Ressources NLTK  — identiques au notebook corrigé
# ============================================================
@st.cache_resource
def load_nltk_resources():
    nltk.download("stopwords",  quiet=True)
    nltk.download("punkt",      quiet=True)
    nltk.download("punkt_tab",  quiet=True)   # requis par word_tokenize (NLTK ≥ 3.9)
    nltk.download("wordnet",    quiet=True)
    nltk.download("omw-1.4",    quiet=True)

load_nltk_resources()

# ============================================================
# Chargement du modèle et du vectorizer
# ============================================================
@st.cache_resource
def load_model():
    missing = [f for f in ("tfidf_vectorizer.pkl", "sentiment_model.pkl")
               if not os.path.exists(f)]
    if missing:
        st.error(
            f"Fichier(s) introuvable(s) : **{', '.join(missing)}**\n\n"
            "Placez `tfidf_vectorizer.pkl` et `sentiment_model.pkl` "
            "dans le même dossier que `app.py`, puis relancez l'application."
        )
        st.stop()
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("sentiment_model.pkl", "rb") as f:
        mdl = pickle.load(f)
    return vectorizer, mdl

tfidf_vectorizer, model = load_model()

# ============================================================
# Prétraitement  — copie exacte du notebook corrigé
# ============================================================
@st.cache_resource
def build_preprocessor():
    sw = set(stopwords.words("english"))
    negation_words = {"not", "no", "never", "n't", "nor", "neither","don't","i don't like","hate"}
    sw = sw - negation_words          # conserver les négations (critique !)
    lm = WordNetLemmatizer()
    return sw, lm

_stop_words, _lemmatizer = build_preprocessor()

def clean_text(text: str) -> str:
    """
    Pipeline identique à celui du notebook corrigé :
      1. Minuscules
      2. Suppression des chiffres
      3. Tokenisation AVANT suppression de la ponctuation
         → préserve "don't" → ["do", "n't"] pour capter les négations
      4. Filtrage des stopwords  (n't conservé)
      5. Suppression de la ponctuation résiduelle token par token
      6. Lemmatisation
    """
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)

    tokens = nltk.word_tokenize(text)                                    # étape 3
    tokens = [w for w in tokens if w not in _stop_words]                # étape 4
    tokens = [w.translate(str.maketrans("", "", string.punctuation))    # étape 5
              for w in tokens]
    tokens = [w for w in tokens if w]                                   # retirer tokens vides
    tokens = [_lemmatizer.lemmatize(w) for w in tokens]                 # étape 6

    return " ".join(tokens)

# ============================================================
# Détection de langue (heuristique légère)
# ============================================================
_FRENCH_MARKERS = {
    "je", "j'aime", "bien", "très", "produit", "mais", "pas", "bon",
    "mauvais", "super", "vraiment", "jamais", "aime", "déteste",
    "service", "livraison", "commande", "qualité", "prix", "achat",
}

def is_likely_non_english(text: str) -> bool:
    words = set(re.findall(r"[a-zéèêëàâùûüîïôœç']+", text.lower()))
    return len(words & _FRENCH_MARKERS) >= 2

# ============================================================
# Score de confiance  (LinearSVC → sigmoïde sur decision_function)
# ============================================================
def confidence_score(review_vector) -> float | None:
    if hasattr(model, "decision_function"):
        raw = model.decision_function(review_vector)[0]
        return 1.0 / (1.0 + math.exp(-abs(raw)))   # sigmoïde : ∈ (0.5, 1.0)
    return None

# ============================================================
# Interface
# ============================================================
st.title("🛒 Analyse de sentiments sur avis clients")
st.write(
    "Prédit automatiquement si un avis client Amazon est **positif** ou **négatif**."
)
st.markdown("---")

review = st.text_area(
    "Entrez un avis client :",
    placeholder="Exemple : The delivery was fast and the product quality is excellent.",
    height=150,
)

col_btn, col_clear, _ = st.columns([1.2, 1.2, 5])
analyse  = col_btn.button("Analyser",  type="primary")
effacer  = col_clear.button("Effacer")

if effacer:
    st.rerun()

if analyse:
    # --- Validation de la saisie ---
    if not review.strip():
        st.warning("Veuillez entrer un avis avant de lancer l'analyse.")
        st.stop()

    # --- Avertissement de langue ---
    if is_likely_non_english(review):
        st.warning(
            "Le modèle a été entraîné sur des avis en **anglais** (Amazon). "
            "Un avis dans une autre langue risque d'être mal classifié.\n\n"
            "Exemple en anglais : *I really like this product.*"
        )

    # --- Nettoyage ---
    cleaned = clean_text(review)

    if not cleaned.strip():
        st.warning(
            "Le texte est vide après nettoyage (chiffres et ponctuation uniquement). "
            "Veuillez entrer un avis plus détaillé."
        )
        st.stop()

    # --- Prédiction ---
    vector     = tfidf_vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    conf       = confidence_score(vector)
    conf_str   = f" — Confiance : **{conf:.0%}**" if conf is not None else ""

    # --- Affichage du résultat ---
    st.markdown("### Résultat")
    if prediction == 1:
        st.success(f"Sentiment prédit : **Positif** 😊{conf_str}")
    else:
        st.error(f"Sentiment prédit : **Négatif** 😞{conf_str}")

    # --- Texte nettoyé (détail optionnel) ---
    with st.expander("Texte après nettoyage"):
        st.code(cleaned, language=None)