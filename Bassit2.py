import os
import warnings
import logging
import base64
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import torch
import numpy as np

# -----------------------------------------
# Image Helper
# -----------------------------------------
def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

# -----------------------------------------
# Streamlit Config
# -----------------------------------------
st.set_page_config(
    page_title="بَيِّنْ وَ بَسِيطْ",
    page_icon="📖",
    layout="centered"
)

# -----------------------------------------
# Arabic Normalization
# -----------------------------------------
ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")

def normalize_ar(text):
    text = str(text)
    text = ARABIC_DIACRITICS.sub("", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[ؤئ]", "ء", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------------------
# Load Models
# -----------------------------------------
@st.cache_resource
def load_classifier():
    tokenizer = AutoTokenizer.from_pretrained("SarahAlhalees/AraBERTv2_RefinedBayyin", use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained("SarahAlhalees/AraBERTv2_RefinedBayyin")
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_simplifier():
    try:
        tokenizer = AutoTokenizer.from_pretrained("SarahAlhalees/bassit-simplifier")
        model     = AutoModelForSeq2SeqLM.from_pretrained("SarahAlhalees/bassit-simplifier")
        return tokenizer, model
    except:
        return None, None

classifier_tokenizer, classifier_model = load_classifier()
simplifier_tokenizer, simplifier_model = load_simplifier()

def classify(text):
    inputs = classifier_tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        logits = classifier_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().numpy()
    prediction = int(np.argmax(probs)) + 1
    confidence = float(probs[prediction - 1])
    return prediction, confidence

# -----------------------------------------
# Simplification helper
# -----------------------------------------
LEVEL_TOKEN = {"mild": "[L3]", "medium": "[L2]", "strong": "[L1]"}
LEVEL_LABELS = {"mild": "تبسيط خفيف", "medium": "تبسيط متوسط", "strong": "تبسيط قوي"}
LEVEL_ICONS = {"mild": "✦", "medium": "✦✦", "strong": "✦✦✦"}

def simplify(text, level_key):
    prefix  = LEVEL_TOKEN[level_key]
    cleaned = normalize_ar(text)
    source  = f"{prefix} {cleaned}"
    inputs  = simplifier_tokenizer(source, return_tensors="pt", truncation=True, max_length=256)
    outputs = simplifier_model.generate(
        **inputs,
        max_length=512,
        num_beams=4,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        min_length=10,
    )
    return simplifier_tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------------------
# UI Styling
# -----------------------------------------
logo_b64 = get_image_base64("logo4.png")
bg_b64   = get_image_base64("jamal.jpg")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Cairo:wght@300;400;700&display=swap');

.stApp {{
    background-image: url("data:image/jpeg;base64,{bg_b64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background: linear-gradient(180deg, rgba(10, 25, 40, 0.75) 0%, rgba(20, 35, 45, 0.90) 100%);
    z-index: 0;
}}

.block-container {{
    position: relative;
    z-index: 1;
    max-width: 800px;
    padding-top: 2rem;
}}

h1, h2, h3, p, span, label {{
    font-family: 'Cairo', sans-serif !important;
    text-align: right !important;
    direction: rtl !important;
    color: #F5EEDC !important;
}}

.logo-wrapper {{
    display: flex;
    justify-content: center;
    margin-top: 5rem !important;
    margin-bottom: 1rem;
}}
.logo-wrapper img {{
    height: 180px;
    filter: drop-shadow(0 0 15px rgba(197, 160, 89, 0.4));
}}

.app-subtitle {{
    text-align: center !important;
    font-family: 'Amiri', serif !important;
    font-size: 1.4rem;
    color: #FFFFFF !important;
    margin-bottom: 2rem;
}}

.gold-divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, #C5A059, transparent);
    margin: 2rem auto;
    width: 50%;
    opacity: 0.6;
}}

textarea {{
    direction: rtl !important;
    text-align: right !important;
    background: rgba(15, 30, 45, 0.80) !important;
    border: 1px solid rgba(197, 160, 89, 0.4) !important;
    border-radius: 15px !important;
    color: #F5EEDC !important;
}}

/* --- BUTTON CENTERING LOGIC --- */
.stButton {{
    display: flex;
    justify-content: center;
}}

.stButton > button {{
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    font-family: 'Cairo', sans-serif !important;
    width: 100% !important;
    max-width: 200px;
}}

/* Force Horizontal Layout for buttons on Mobile */
[data-testid="stHorizontalBlock"] {{
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 10px !important;
}}

[data-testid="stHorizontalBlock"] > div {{
    flex: 0 1 auto !important;
    min-width: 110px !important; /* Prevents them from getting too small */
}}

.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #C5A059 0%, #8E733E 100%) !important;
    color: #0E1E2B !important;
    font-weight: 700;
    height: 3.5rem;
}}

.stButton > button[kind="secondary"], 
.stButton > button:not([kind="primary"]) {{
    background: rgba(12, 24, 36, 0.88) !important;
    color: #F5EEDC !important;
    border: 1.5px solid rgba(197, 160, 89, 0.65) !important;
    font-size: 0.9rem !important;
}}

.stat-pill {{
    display: block;
    padding: 12px;
    border-radius: 10px;
    background: rgba(8, 18, 30, 0.82);
    border: 1px solid rgba(197, 160, 89, 0.35);
    text-align: center;
    margin: 10px auto;
    width: fit-content;
    min-width: 200px;
}}

.simplified-box {{
    background: rgba(8, 18, 28, 0.88);
    padding: 20px;
    border-radius: 15px;
    border-right: 5px solid #C5A059;
    margin-top: 15px;
    direction: rtl;
    text-align: right;
}}

@media (max-width: 480px) {{
    .stButton > button {{
        font-size: 0.8rem !important;
        height: 3rem !important;
        padding: 0 4px !important;
    }}
    .app-subtitle {{ font-size: 1.1rem; }}
}}
</style>

<div class="logo-wrapper">
    <img src="data:image/png;base64,{logo_b64}">
</div>
<div class="app-subtitle">نظام ذكي لتصنيف مستوى مقروئية النصوص العربية وتبسيطها</div>
<div class="gold-divider"></div>
""", unsafe_allow_html=True)

# -----------------------------------------
# Main Logic
# -----------------------------------------
text = st.text_area("أدخل النص المراد تصنيفه:", height=220, placeholder="اكتب أو الصق النص هنا...")

col_b1, col_b2, col_b3 = st.columns([1, 1, 1])
with col_b2:
    if st.button("بَيِّنْ", type="primary"):
        if text.strip():
            # Basic validation
            if not re.search(r"[\u0600-\u06FF]", text):
                st.error("الرجاء إدخال نص باللغة العربية")
            else:
                with st.spinner("جاري التصنيف..."):
                    cleaned = normalize_ar(text)
                    level, conf = classify(cleaned)
                    st.session_state.done = True
                    st.session_state.level = level
                    st.session_state.text = text
                    st.session_state.simplified_results = {}
        else:
            st.error("الرجاء إدخال نص")

if st.session_state.get("done"):
    st.markdown(f"<div class='stat-pill'>مستوى الصعوبة: <b style='color:#C5A059; font-size:1.4rem;'>{st.session_state.level}</b></div>", unsafe_allow_html=True)

    if st.session_state.level >= 4:
        st.markdown("<p style='text-align:center !important;'>اختر درجة التبسيط المطلوبة:</p>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("تبسيط خفيف ✦", key="btn_mild"):
                st.session_state.simplified_results["mild"] = simplify(st.session_state.text, "mild")
        with c2:
            if st.button("تبسيط متوسط ✦✦", key="btn_medium"):
                st.session_state.simplified_results["medium"] = simplify(st.session_state.text, "medium")
        with c3:
            if st.button("تبسيط قوي ✦✦✦", key="btn_strong"):
                st.session_state.simplified_results["strong"] = simplify(st.session_state.text, "strong")

        for k in ("mild", "medium", "strong"):
            res = st.session_state.simplified_results.get(k)
            if res:
                st.markdown(f"""
                <div class='simplified-box'>
                    <b style='color:#C5A059;'>{LEVEL_ICONS[k]} النتيجة ({LEVEL_LABELS[k]}):</b><br>{res}
                </div>
                """, unsafe_allow_html=True)

st.markdown('<div style="text-align: center; color: #C5A059; margin-top: 50px; font-size: 0.8rem; opacity: 0.6;">© 2026 — مشروع بَيِّنْ وَ بَسِيطْ</div>', unsafe_allow_html=True)
