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
# UI Styling
# -----------------------------------------
logo_b64 = get_image_base64("logo4.png")
bg_b64   = get_image_base64("jamal.jpg")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Cairo:wght@300;400;700&display=swap');

/* Background Overlay */
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
    background: linear-gradient(180deg, rgba(10, 25, 40, 0.7) 0%, rgba(20, 35, 45, 0.85) 100%);
    z-index: 0;
}}

.block-container {{
    position: relative;
    z-index: 1;
    max-width: 800px;
    padding-top: 2rem;
}}

/* Typography */
h1, h2, h3, p, span, label {{
    font-family: 'Cairo', sans-serif !important;
    text-align: right !important;
    direction: rtl !important;
    color: #F5EEDC !important;
}}

/* Logo Animation */
.logo-wrapper {{
    display: flex;
    justify-content: center;
    margin-top: 5rem !important; 
    margin-bottom: 1rem;
    animation: fadeInDown 1.5s ease-out;
}}
.logo-wrapper img {{
    height: 180px;
    filter: drop-shadow(0 0 15px rgba(197, 160, 89, 0.4));
}}

@keyframes fadeInDown {{
    from {{ opacity: 0; transform: translateY(-20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

/* SUBTITLE CHANGED TO WHITE */
.app-subtitle {{
    text-align: center !important;
    font-family: 'Amiri', serif !important;
    font-size: 1.4rem;
    color: #FFFFFF !important;
    margin-bottom: 2rem;
    letter-spacing: 1px;
}}

/* Elegant Divider */
.gold-divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, #C5A059, transparent);
    margin: 2rem auto;
    width: 50%;
    opacity: 0.6;
}}

/* Input Box */
textarea {{
    direction: rtl !important;
    text-align: right !important;
    background: rgba(15, 30, 45, 0.75) !important;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    color: #F5EEDC !important;
    border: 1px solid rgba(197, 160, 89, 0.3) !important;
    border-radius: 15px !important;
    padding: 15px !important;
    font-size: 1.1rem !important;
    caret-color: #F5EEDC !important;
}}

textarea::placeholder {{
    color: rgba(245, 238, 220, 0.45) !important;
}}

/* Force text color even when Streamlit overrides in light mode */
div[data-testid="stTextArea"] textarea,
div[data-baseweb="textarea"] textarea,
.stTextArea textarea {{
    color: #F5EEDC !important;
    background: rgba(15, 30, 45, 0.75) !important;
}}

/* BUTTON CENTERING CSS */
.stButton {{
    display: flex;
    justify-content: center;
}}

.stButton > button {{
    width: 200px !important; /* Fixed width for a centered appearance */
    border-radius: 12px !important;
    border: none !important;
    height: 3.5rem;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    transition: all 0.3s ease !important;
}}

.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #C5A059 0%, #8E733E 100%) !important;
    color: #14232D !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}}

.stButton > button:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(197, 160, 89, 0.3) !important;
}}

/* Result Cards */
.simplified-box {{
    background: rgba(197, 160, 89, 0.1);
    backdrop-filter: blur(5px);
    padding: 25px;
    border-radius: 15px;
    color: #F5EEDC;
    border-right: 5px solid #C5A059;
    margin-top: 25px;
    line-height: 1.8;
    font-size: 1.1rem;
}}

/* Progress Bar */
div[data-testid="stProgress"] > div > div > div > div {{
    background-color: #C5A059 !important;
}}
</style>

<div class="logo-wrapper">
    <img src="data:image/png;base64,{logo_b64}">
</div>
<div class="app-subtitle">نظام ذكي لتصنيف مستوى مقروئية النصوص العربية وتبسيطها</div>
<div class="gold-divider"></div>
""", unsafe_allow_html=True)

# -----------------------------------------
# Content
# -----------------------------------------
text = st.text_area("أدخل النص المراد تصنيفه:", height=220, placeholder="اكتب أو الصق النص هنا...")

if 'done' not in st.session_state:
    st.session_state.done = False

# Centering the 'Bayyin' button using columns
col_b1, col_b2, col_b3 = st.columns([1, 1, 1])
with col_b2:
    if st.button("بَيِّنْ", type="primary"):
        if text.strip():
            with st.spinner('يتم الآن فحص لغة النص...'):
                cleaned = normalize_ar(text)
                level, conf = classify(cleaned)
                st.session_state.done = True
                st.session_state.level = level
                st.session_state.conf = conf
                st.session_state.text = text
        else:
            st.error("الرجاء تزويدنا بنص للبدء")

if st.session_state.done:
    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col2:
        st.markdown(f"<div style='font-size: 1.2rem;'>مستوى الصعوبة: <span style='color:#C5A059; font-weight:bold;'>{st.session_state.level}</span></div>", unsafe_allow_html=True)
    with col1:
        st.markdown(f"<div style='text-align: left; opacity:0.8;'>الدقة: {st.session_state.conf:.1%}</div>", unsafe_allow_html=True)
    
    st.progress(st.session_state.conf)

    if st.session_state.level >= 4:
        st.markdown("<br>", unsafe_allow_html=True)
        # Centering the 'Bassit' button using columns
        col_s1, col_s2, col_s3 = st.columns([1, 1, 1])
        with col_s2:
            if st.button("بَسِّطْ"):
                if simplifier_model:
                    with st.spinner('جاري إعادة صياغة النص بأسلوب أبسط...'):
                        cleaned = normalize_ar(st.session_state.text)
                        inputs = simplifier_tokenizer(cleaned, return_tensors="pt")
                        outputs = simplifier_model.generate(**inputs, max_length=512)
                        simplified = simplifier_tokenizer.decode(outputs[0], skip_special_tokens=True)

                        st.markdown(f"""
                        <div class='simplified-box'>
                            <strong style='color:#C5A059'>النتيجة المبسطة:</strong><br>
                            {simplified}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("خدمة التبسيط غير متاحة حالياً")

st.markdown("""
<div style="text-align: center; color: #C5A059; margin-top: 60px; font-size: 0.85rem; opacity: 0.6; font-family: 'Cairo';">
    © 2026 — مشروع بَيِّنْ وَ بَسِّطْ
</div>
""", unsafe_allow_html=True)
