import os
import re
import math
import string
import pickle
import time
from collections import Counter

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="SentimentIQ · Analyse d'avis clients",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS personnalisé — design sombre & élégant
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Fond principal */
.stApp {
    background: #0f1117;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #1e2d40;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #94a3b8;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}

/* Titres */
h1 { font-family: 'Space Grotesk', sans-serif !important; }
h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

/* Cards custom */
.metric-card {
    background: #161b27;
    border: 1px solid #1e2d40;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}

.result-positive {
    background: linear-gradient(135deg, #0a2e1a 0%, #0d3320 100%);
    border: 1px solid #166534;
    border-left: 4px solid #22c55e;
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
}

.result-negative {
    background: linear-gradient(135deg, #2d0a0a 0%, #3a0f0f 100%);
    border: 1px solid #991b1b;
    border-left: 4px solid #ef4444;
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
}

.result-neutral {
    background: linear-gradient(135deg, #1a1d2e 0%, #1e2235 100%);
    border: 1px solid #334155;
    border-left: 4px solid #94a3b8;
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
}

.tag-positive {
    display: inline-block;
    background: #14532d;
    color: #86efac;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    margin: 3px;
    font-weight: 500;
}

.tag-negative {
    display: inline-block;
    background: #450a0a;
    color: #fca5a5;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    margin: 3px;
    font-weight: 500;
}

.tag-neutral {
    display: inline-block;
    background: #1e293b;
    color: #94a3b8;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    margin: 3px;
}

.history-card {
    background: #161b27;
    border: 1px solid #1e2d40;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    font-size: 14px;
}

.badge-pos {
    background: #166534;
    color: #86efac;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

.badge-neg {
    background: #7f1d1d;
    color: #fca5a5;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

/* Textarea */
.stTextArea textarea {
    background: #161b27 !important;
    border: 1px solid #1e2d40 !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
}
.stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
}

/* Bouton primaire */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 10px 32px !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.4) !important;
}

/* Bouton secondaire */
.stButton > button[kind="secondary"] {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
}

/* Progress bar */
.stProgress .st-bo { background-color: #1e293b; }

/* Selectbox */
.stSelectbox > div > div {
    background: #161b27 !important;
    border-color: #1e2d40 !important;
    color: #e2e8f0 !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #161b27 !important;
    border: 1px solid #1e2d40 !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #161b27;
    border: 1px solid #1e2d40;
    border-radius: 12px;
    padding: 16px;
}

/* Divider */
hr { border-color: #1e2d40 !important; }

/* Slider */
.stSlider .st-bx { background: #3b82f6 !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# NLTK resources
# ============================================================
@st.cache_resource
def load_nltk_resources():
    for res in ["stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"]:
        nltk.download(res, quiet=True)

load_nltk_resources()


# ============================================================
# Chargement modèle
# ============================================================
@st.cache_resource
def load_model():
    missing = [f for f in ("tfidf_vectorizer.pkl", "sentiment_model.pkl")
               if not os.path.exists(f)]
    if missing:
        st.error(
            f"**Fichiers manquants :** `{', '.join(missing)}`\n\n"
            "Placez `tfidf_vectorizer.pkl` et `sentiment_model.pkl` dans le dossier de `app.py`."
        )
        st.stop()
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("sentiment_model.pkl", "rb") as f:
        mdl = pickle.load(f)
    return vectorizer, mdl

tfidf_vectorizer, model = load_model()


# ============================================================
# Prétraitement
# ============================================================
@st.cache_resource
def build_preprocessor():
    sw = set(stopwords.words("english"))
    negation_words = {"not", "no", "never", "n't", "nor", "neither", "don't", "hate"}
    sw = sw - negation_words
    lm = WordNetLemmatizer()
    return sw, lm

_stop_words, _lemmatizer = build_preprocessor()

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in _stop_words]
    tokens = [w.translate(str.maketrans("", "", string.punctuation)) for w in tokens]
    tokens = [w for w in tokens if w]
    tokens = [_lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


# ============================================================
# Utilitaires
# ============================================================
_FRENCH_MARKERS = {
    "je", "bien", "très", "produit", "mais", "pas", "bon", "mauvais",
    "super", "vraiment", "jamais", "aime", "déteste", "livraison",
    "commande", "qualité", "prix", "achat", "article",
}

def is_likely_non_english(text: str) -> bool:
    words = set(re.findall(r"[a-zéèêëàâùûüîïôœç']+", text.lower()))
    return len(words & _FRENCH_MARKERS) >= 2

def confidence_score(review_vector) -> float | None:
    if hasattr(model, "decision_function"):
        raw = model.decision_function(review_vector)[0]
        return 1.0 / (1.0 + math.exp(-abs(raw)))
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(review_vector)[0]
        return float(max(proba))
    return None

# Mots clés positifs / négatifs simples
_POS_WORDS = {"good","great","excellent","amazing","love","perfect","best","awesome",
              "fantastic","wonderful","happy","satisfied","recommend","quality","fast",
              "nice","beautiful","easy","helpful","friendly","superb","outstanding"}
_NEG_WORDS = {"bad","terrible","awful","horrible","worst","hate","poor","broken",
              "useless","disappointing","slow","waste","cheap","defective","wrong",
              "problem","issue","return","refund","damaged","fake","scam","never"}

def extract_keywords(text: str) -> tuple[list, list]:
    words = re.findall(r"[a-z]+", text.lower())
    pos = [w for w in words if w in _POS_WORDS]
    neg = [w for w in words if w in _NEG_WORDS]
    return list(dict.fromkeys(pos)), list(dict.fromkeys(neg))  # dédupliqués, ordre préservé

def get_intensity_label(conf: float) -> tuple[str, str]:
    if conf is None:
        return "Inconnu", "#94a3b8"
    if conf >= 0.90:
        return "Très fort", "#22c55e"
    elif conf >= 0.75:
        return "Fort", "#84cc16"
    elif conf >= 0.60:
        return "Modéré", "#f59e0b"
    else:
        return "Faible", "#f97316"

def word_count(text: str) -> int:
    return len(text.split())


# ============================================================
# État de session
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "total_analyzed" not in st.session_state:
    st.session_state.total_analyzed = 0
if "total_positive" not in st.session_state:
    st.session_state.total_positive = 0


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <div style="font-family:'Space Grotesk',sans-serif; font-size:22px; font-weight:700;
                    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            SentimentIQ
        </div>
        <div style="color:#475569; font-size:12px; margin-top:4px;">Analyse d'avis clients Amazon</div>
    </div>
    <hr style="margin: 8px 0 20px;">
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Paramètres")
    
    show_keywords = st.toggle("Afficher les mots-clés détectés", value=True)
    show_cleaned = st.toggle("Afficher le texte nettoyé", value=False)
    show_history = st.toggle("Afficher l'historique", value=True)
    
    st.markdown("---")
    st.markdown("### 📝 Exemples rapides")
    
    examples = {
        "⭐ Positif court": "Fast delivery, great quality product. Very satisfied!",
        "⭐ Positif détaillé": "Absolutely love this product! The quality is outstanding, delivery was super fast and the customer service was incredibly helpful. I highly recommend it.",
        "💔 Négatif court": "Total waste of money. Broke after 2 days.",
        "💔 Négatif détaillé": "This is the worst product I've ever bought. It was damaged on arrival, the customer service never responded, and it stopped working after 3 uses. Do not buy!",
        "🤔 Mitigé": "The product is okay, not great not terrible. Delivery was slow but packaging was fine.",
    }
    
    selected_example = st.selectbox(
        "Choisir un exemple :",
        ["— Sélectionner —"] + list(examples.keys()),
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Stats globales
    if st.session_state.total_analyzed > 0:
        st.markdown("### 📊 Session actuelle")
        pct = (st.session_state.total_positive / st.session_state.total_analyzed) * 100
        
        col1, col2 = st.columns(2)
        col1.metric("Analysés", st.session_state.total_analyzed)
        col2.metric("Positifs", f"{pct:.0f}%")
        
        st.progress(pct / 100)
    
    st.markdown("---")
    if st.button("🗑️ Effacer l'historique", use_container_width=True):
        st.session_state.history = []
        st.session_state.total_analyzed = 0
        st.session_state.total_positive = 0
        st.rerun()


# ============================================================
# CONTENU PRINCIPAL
# ============================================================

# Header
st.markdown("""
<div style="padding: 8px 0 24px;">
    <h1 style="font-family:'Space Grotesk',sans-serif; font-size:32px; font-weight:700;
               margin:0; color:#f1f5f9;">
        🔍 Analyse de Sentiments
    </h1>
    <p style="color:#64748b; margin:6px 0 0; font-size:15px;">
        Classifiez automatiquement les avis clients Amazon comme positifs ou négatifs
    </p>
</div>
""", unsafe_allow_html=True)

# Colonnes principales
col_main, col_info = st.columns([3, 1], gap="large")

with col_main:
    # Pré-remplir depuis un exemple
    default_text = ""
    if selected_example != "— Sélectionner —":
        default_text = examples[selected_example]
    
    review = st.text_area(
        "Avis client (en anglais) :",
        value=default_text,
        placeholder="Exemple : The delivery was fast and the product quality is excellent. I'll definitely order again!",
        height=160,
        label_visibility="visible",
    )
    
    # Compteurs en temps réel
    wc = word_count(review) if review.strip() else 0
    char_c = len(review)
    
    st.markdown(
        f'<div style="text-align:right; color:#475569; font-size:12px; margin-top:-8px;">'
        f'{wc} mots · {char_c} caractères</div>',
        unsafe_allow_html=True
    )
    
    btn_col1, btn_col2, btn_col3 = st.columns([2, 1.5, 4])
    with btn_col1:
        analyse = st.button("🔍 Analyser", type="primary", use_container_width=True)
    with btn_col2:
        effacer = st.button("✕ Effacer", use_container_width=True)

with col_info:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size:11px; color:#475569; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px; font-weight:600;">
            Comment ça fonctionne
        </div>
        <div style="font-size:13px; color:#94a3b8; line-height:1.7;">
            <b style="color:#cbd5e1">1.</b> Saisissez un avis<br>
            <b style="color:#cbd5e1">2.</b> Nettoyage du texte<br>
            <b style="color:#cbd5e1">3.</b> Vectorisation TF-IDF<br>
            <b style="color:#cbd5e1">4.</b> Classification ML<br>
            <b style="color:#cbd5e1">5.</b> Score de confiance
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-card" style="margin-top:0;">
        <div style="font-size:11px; color:#475569; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px; font-weight:600;">
            Conseils
        </div>
        <div style="font-size:12px; color:#64748b; line-height:1.7;">
            ✦ Rédigez en <b style="color:#94a3b8">anglais</b><br>
            ✦ Minimum <b style="color:#94a3b8">5 mots</b><br>
            ✦ Soyez précis<br>
            ✦ Évitez les abréviations
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Traitement
# ============================================================
if effacer:
    st.rerun()

if analyse:
    if not review.strip():
        st.warning("⚠️ Veuillez entrer un avis avant de lancer l'analyse.")
        st.stop()
    
    if wc < 2:
        st.warning("⚠️ L'avis est trop court. Veuillez entrer au moins quelques mots.")
        st.stop()

    # Avertissement langue
    if is_likely_non_english(review):
        st.warning(
            "🌍 **Langue détectée :** Le modèle est entraîné sur des avis en anglais. "
            "Un texte dans une autre langue peut donner des résultats incorrects."
        )

    # Animation de chargement
    with st.spinner("Analyse en cours…"):
        time.sleep(0.4)
        cleaned = clean_text(review)

    if not cleaned.strip():
        st.warning("⚠️ Texte vide après nettoyage. Veuillez entrer un avis plus descriptif.")
        st.stop()

    # Prédiction
    vector = tfidf_vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    conf = confidence_score(vector)
    conf_pct = f"{conf:.0%}" if conf is not None else "N/A"
    intensity_label, intensity_color = get_intensity_label(conf)
    
    # Mots-clés
    pos_kw, neg_kw = extract_keywords(review)

    # Mise à jour stats
    st.session_state.total_analyzed += 1
    if prediction == 1:
        st.session_state.total_positive += 1

    # Ajout historique
    st.session_state.history.insert(0, {
        "text": review[:80] + ("…" if len(review) > 80 else ""),
        "prediction": prediction,
        "confidence": conf,
        "words": wc,
    })
    if len(st.session_state.history) > 10:
        st.session_state.history.pop()

    st.markdown("---")

    # ---- Résultat principal ----
    if prediction == 1:
        emoji = "😊"
        label = "Positif"
        css_class = "result-positive"
        accent = "#22c55e"
    else:
        emoji = "😞"
        label = "Négatif"
        css_class = "result-negative"
        accent = "#ef4444"

    st.markdown(f"""
    <div class="{css_class}">
        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
            <div>
                <div style="font-size:13px; color:#94a3b8; margin-bottom:4px; text-transform:uppercase; letter-spacing:1px; font-weight:600;">
                    Sentiment Détecté
                </div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:28px; font-weight:700; color:#f1f5f9;">
                    {emoji} {label}
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:13px; color:#94a3b8; margin-bottom:4px; text-transform:uppercase; letter-spacing:1px; font-weight:600;">
                    Confiance
                </div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:28px; font-weight:700; color:{accent};">
                    {conf_pct}
                </div>
                <div style="font-size:12px; color:{intensity_color}; margin-top:2px;">
                    Intensité : {intensity_label}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Barre de confiance ----
    if conf is not None:
        st.markdown(
            f'<div style="color:#475569; font-size:12px; margin-bottom:4px;">Score de confiance : {conf_pct}</div>',
            unsafe_allow_html=True
        )
        st.progress(conf)

    # ---- Colonnes détails ----
    det1, det2, det3 = st.columns(3)
    det1.metric("Mots analysés", wc)
    det2.metric("Confiance", conf_pct)
    det3.metric("Prédiction", label)

    # ---- Mots-clés détectés ----
    if show_keywords and (pos_kw or neg_kw):
        st.markdown("#### 🏷️ Mots-clés détectés")
        tags_html = ""
        for w in pos_kw:
            tags_html += f'<span class="tag-positive">✓ {w}</span>'
        for w in neg_kw:
            tags_html += f'<span class="tag-negative">✗ {w}</span>'
        if not pos_kw and not neg_kw:
            tags_html = '<span class="tag-neutral">Aucun mot-clé connu trouvé</span>'
        st.markdown(f'<div style="margin-top:8px;">{tags_html}</div>', unsafe_allow_html=True)

    # ---- Texte nettoyé ----
    if show_cleaned:
        with st.expander("🔧 Texte après prétraitement NLP"):
            st.code(cleaned, language=None)
            tok_count = len(cleaned.split())
            st.caption(f"{tok_count} tokens après nettoyage (stopwords, ponctuation, lemmatisation)")

    # ---- Analyse de nuance ----
    if pos_kw and neg_kw:
        st.info(
            "💡 **Avis mixte détecté** : Ce texte contient à la fois des termes positifs "
            f"({', '.join(pos_kw[:3])}) et négatifs ({', '.join(neg_kw[:3])}). "
            "La prédiction finale reflète le sentiment dominant."
        )

    # ---- Réanalyser un exemple similaire ----
    st.markdown(
        '<div style="color:#475569; font-size:13px; margin-top:16px;">'
        'Modifiez l\'avis ci-dessus et cliquez à nouveau sur Analyser pour comparer.</div>',
        unsafe_allow_html=True
    )


# ============================================================
# Historique
# ============================================================
if show_history and st.session_state.history:
    st.markdown("---")
    st.markdown("#### 🕐 Historique de la session")
    
    for i, item in enumerate(st.session_state.history):
        badge = '<span class="badge-pos">Positif</span>' if item["prediction"] == 1 \
                else '<span class="badge-neg">Négatif</span>'
        conf_str = f"{item['confidence']:.0%}" if item["confidence"] else "N/A"
        
        st.markdown(f"""
        <div class="history-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px;">
                <div style="color:#cbd5e1; flex:1; font-size:14px; line-height:1.5;">
                    {item['text']}
                </div>
                <div style="display:flex; flex-direction:column; align-items:flex-end; gap:4px; min-width:80px;">
                    {badge}
                    <span style="color:#475569; font-size:12px;">{conf_str} · {item['words']} mots</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#334155; font-size:12px; padding:12px 0 24px;">
    SentimentIQ · Modèle TF-IDF + LinearSVC · Entraîné sur avis Amazon (anglais)
</div>
""", unsafe_allow_html=True)
