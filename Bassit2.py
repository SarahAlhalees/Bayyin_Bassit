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
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# -----------------------------------------
# Streamlit Config
# -----------------------------------------
st.set_page_config(
    page_title="بَيِّنْ - تصنيف وتبسيط النصوص العربية",
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
        st.warning(f"نموذج التبسيط غير متوفر: {str(e)}")
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
# UI Styling
# -----------------------------------------
logo_b64 = get_image_base64("logo2.png")
bg_b64   = get_image_base64("jamal.jpg")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Cairo:wght@400;600;700&display=swap');

#MainMenu, footer, header {{visibility: hidden;}}

.stApp {{
    background-image: url("data:image/jpeg;base64,{bg_b64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'Cairo', sans-serif;
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
}}

.logo-wrapper {{
    display: flex;
    justify-content: center;
    margin: 2.5rem auto 1rem auto;
    animation: fadeIn 1.2s ease;
}}

.logo-wrapper img {{
    height: 220px;
    max-width: 90%;
    object-fit: contain;
    filter: drop-shadow(0 6px 20px rgba(212,175,55,0.5));
}}

@media (max-width: 768px) {{
    .logo-wrapper img {{
        height: 160px;
    }}
}}

@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(-10px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.app-subtitle {{
    text-align: center;
    color: #e8dfc8;
    margin-bottom: 1rem;
}}

.gold-divider {{
    height: 2px;
    background: linear-gradient(90deg, transparent, #D4AF37, transparent);
    margin: 1rem auto;
    width: 60%;
}}

textarea {{
    direction: rtl;
    text-align: right;
    background: rgba(10,25,60,0.75);
    color: #f0e6c8;
    border: 1.5px solid #D4AF37;
    border-radius: 10px;
}}

.stButton > button[kind="primary"] {{
    background: #D4AF37;
    color: #0a1940;
    font-size: 1.2rem;
    border-radius: 10px;
}}

.simplified-box {{
    background: rgba(10,25,60,0.85);
    padding: 20px;
    border-radius: 12px;
    color: #f0e6c8;
    border-right: 4px solid #D4AF37;
    margin-top: 20px;
}}
</style>

<div class="logo-wrapper">
    <img src="data:image/png;base64,{logo_b64}">
</div>

<div class="app-subtitle">مصنِّف وتبسيط النصوص العربية</div>
<div class="gold-divider"></div>
""", unsafe_allow_html=True)

# -----------------------------------------
# Input
# -----------------------------------------
text = st.text_area("أدخل النص العربي", height=200)

# Session state
if 'done' not in st.session_state:
    st.session_state.done = False

# -----------------------------------------
# Classify Button
# -----------------------------------------
if st.button("تحليل", use_container_width=True):
    if text.strip():
        cleaned = normalize_ar(text)
        level, conf = classify(cleaned)

        st.session_state.done = True
        st.session_state.level = level
        st.session_state.conf = conf
        st.session_state.text = text
    else:
        st.warning("الرجاء إدخال نص")

# -----------------------------------------
# Results
# -----------------------------------------
if st.session_state.done:
    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)

    level = st.session_state.level
    conf  = st.session_state.conf

    st.write(f"المستوى: {level}")
    st.progress(int(conf * 100))
    st.write(f"الثقة: {conf:.2%}")

    if level >= 4:
        if st.button("تبسيط"):
            if simplifier_model:
                cleaned = normalize_ar(st.session_state.text)
                inputs = simplifier_tokenizer(cleaned, return_tensors="pt")

                outputs = simplifier_model.generate(**inputs)
                simplified = simplifier_tokenizer.decode(outputs[0], skip_special_tokens=True)

                st.markdown(f"<div class='simplified-box'>{simplified}</div>", unsafe_allow_html=True)
            else:
                st.warning("نموذج التبسيط غير متوفر")

st.caption("© 2025 — مشروع بَيِّنْ")
