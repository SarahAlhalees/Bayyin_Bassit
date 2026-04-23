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
    page_title="بَيِّنْ وَ بَسِّطْ - تصنيف وتبسيط النصوص العربية",
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
    tokenizer = AutoTokenizer.from_pretrained(
        "SarahAlhalees/AraBERTv2_RefinedBayyin",
        use_fast=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "SarahAlhalees/AraBERTv2_RefinedBayyin"
    )
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_simplifier():
    try:
        tokenizer = AutoTokenizer.from_pretrained("SarahAlhalees/bassit-simplifier")
        model     = AutoModelForSeq2SeqLM.from_pretrained("SarahAlhalees/bassit-simplifier")
        return tokenizer, model
    except Exception as e:
        return None, None

classifier_tokenizer, classifier_model = load_classifier()
simplifier_tokenizer, simplifier_model = load_simplifier()

# -----------------------------------------
# Inference
# -----------------------------------------
def classify(text):
    inputs = classifier_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )
    with torch.no_grad():
        logits = classifier_model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze().numpy()
    prediction = int(np.argmax(probs)) + 1
    confidence = float(probs[prediction - 1])
    return prediction, confidence

# -----------------------------------------
# UI Styling & RTL Alignment
# -----------------------------------------
logo_b64 = get_image_base64("logo3.png")
bg_b64   = get_image_base64("jamal.jpg")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Cairo:wght@400;600;700&display=swap');

/* Main App Layout */
.stApp {{
    background-image: url("data:image/jpeg;base64,{bg_b64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'Cairo', sans-serif;
    direction: rtl; /* Global Right-to-Left */
}}

.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(10,25,60,0.8);
    z-index: 0;
}}

.block-container {{
    position: relative;
    z-index: 1;
    max-width: 780px;
    text-align: right;
}}

/* Element Alignment */
h1, h2, h3, p, span, label, .stMarkdown {{
    text-align: right !important;
    direction: rtl !important;
}}

/* Logo */
.logo-wrapper {{
    display: flex;
    justify-content: center;
    margin: 2.5rem auto 1rem auto;
}}

.logo-wrapper img {{
    height: 220px;
    max-width: 90%;
    filter: drop-shadow(0 6px 20px rgba(212,175,55,0.5));
}}

.app-subtitle {{
    text-align: center !important;
    color: #e8dfc8;
    margin-bottom: 1rem;
}}

.gold-divider {{
    height: 2px;
    background: linear-gradient(90deg, transparent, #D4AF37, transparent);
    margin: 1.5rem auto;
    width: 60%;
}}

/* Input Box */
textarea {{
    direction: rtl !important;
    text-align: right !important;
    background: rgba(10,25,60,0.75) !important;
    color: #f0e6c8 !important;
    border: 1.5px solid #D4AF37 !important;
    border-radius: 10px !important;
}}

/* Progress Bar Alignment */
div[data-testid="stProgress"] {{
    direction: rtl;
}}

/* Result Cards */
.result-text {{
    color: #f0e6c8;
    font-size: 1.2rem;
    margin-bottom: 5px;
}}

.simplified-box {{
    background: rgba(10,25,60,0.85);
    padding: 20px;
    border-radius: 12px;
    color: #f0e6c8;
    border-right: 4px solid #D4AF37;
    margin-top: 20px;
    text-align: right;
    direction: rtl;
}}

.stButton > button {{
    width: 100%;
    border-radius: 10px;
}}

.stButton > button[kind="primary"] {{
    background: #D4AF37;
    color: #0a1940;
    font-weight: bold;
}}
</style>

<div class="logo-wrapper">
    <img src="data:image/png;base64,{logo_b64}">
</div>

<div class="app-subtitle">نظام ذكي لتصنيف وتبسيط النصوص العربية</div>
<div class="gold-divider"></div>
""", unsafe_allow_html=True)

# -----------------------------------------
# Input Section
# -----------------------------------------
# Label is automatically handled by the CSS direction: rtl
text = st.text_area("أدخل النص المراد تحليله هنا:", height=200)

if 'done' not in st.session_state:
    st.session_state.done = False

# -----------------------------------------
# Action
# -----------------------------------------
if st.button("تحليل النص", type="primary"):
    if text.strip():
        with st.spinner('جاري التحليل...'):
            cleaned = normalize_ar(text)
            level, conf = classify(cleaned)
            
            st.session_state.done = True
            st.session_state.level = level
            st.session_state.conf = conf
            st.session_state.text = text
    else:
        st.error("الرجاء إدخال نص أولاً")

# -----------------------------------------
# Results Section
# -----------------------------------------
if st.session_state.done:
    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)
    
    level = st.session_state.level
    conf  = st.session_state.conf

    # Custom HTML for right-aligned results
    st.markdown(f"<div class='result-text'>مستوى الصعوبة: {level}</div>", unsafe_allow_html=True)
    st.progress(int(conf * 100))
    st.markdown(f"<div style='text-align: left; color: #D4AF37; font-size: 0.9rem;'>نسبة الثقة: {conf:.2%}</div>", unsafe_allow_html=True)

    if level >= 4:
        st.info("هذا النص يعتبر معقداً، يمكنك محاولة تبسيطه أدناه.")
        if st.button("تبسيط النص"):
            if simplifier_model:
                with st.spinner('جاري التبسيط...'):
                    cleaned = normalize_ar(st.session_state.text)
                    inputs = simplifier_tokenizer(cleaned, return_tensors="pt")
                    outputs = simplifier_model.generate(**inputs, max_length=512)
                    simplified = simplifier_tokenizer.decode(outputs[0], skip_special_tokens=True)

                    st.markdown(f"""
                    <div class='simplified-box'>
                        <strong>النص المبسط:</strong><br>
                        {simplified}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("عذراً، نموذج التبسيط غير متوفر حالياً.")

st.markdown("""
<div style="text-align: center; color: #888; margin-top: 50px; font-size: 0.8rem;">
    © 2025 — مشروع بَيِّنْ وَ بَسِّطْ 
</div>
""", unsafe_allow_html=True)
