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
# UI Styling
# -----------------------------------------
logo_b64 = get_image_base64("logo3.png")
bg_b64   = get_image_base64("jamal.jpg")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Cairo:wght@400;600;700&display=swap');

#MainMenu, footer, header {{visibility: hidden;}}

/* Global RTL and Right-Alignment */
.stApp {{
    background-image: url("data:image/jpeg;base64,{bg_b64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'Cairo', sans-serif;
    direction: rtl;
    text-align: right;
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
    direction: rtl;
    text-align: right;
}}

/* Target all text containers to be right-aligned */
.rtl-wrapper {{
    direction: rtl !important;
    text-align: right !important;
    width: 100%;
}}

.stMarkdown, .stMarkdown p, .stAlert, .stAlert p,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3 {{
    direction: rtl !important;
    text-align: right !important;
    color: #e8dfc8 !important;
}}

/* Input Labels */
label, div[data-testid="stWidgetLabel"] p {{
    direction: rtl !important;
    text-align: right !important;
    color: #D4AF37 !important;
    width: 100% !important;
}}

/* Text Area Alignment */
textarea {{
    direction: rtl !important;
    text-align: right !important;
    background: rgba(10,25,60,0.75) !important;
    color: #f0e6c8 !important;
    border: 1.5px solid #D4AF37 !important;
}}

/* Results Alignment */
[data-testid="metric-container"] {{
    direction: rtl !important;
    text-align: right !important;
    background: rgba(10, 25, 60, 0.75) !important;
    border: 1px solid #D4AF37 !important;
    padding: 10px !important;
}}

.simplified-box {{
    background: rgba(10,25,60,0.85);
    padding: 20px;
    border-radius: 12px;
    border-right: 4px solid #D4AF37;
    direction: rtl;
    text-align: right;
    color: #f0e6c8;
}}

.logo-wrapper {{
    display: flex;
    justify-content: center;
    margin-bottom: 1rem;
}}
.logo-wrapper img {{ height: 200px; filter: drop-shadow(0 4px 10px #D4AF37); }}

.gold-divider {{
    height: 2px;
    background: linear-gradient(90deg, transparent, #D4AF37, transparent);
    margin: 20px 0;
}}
</style>

<div class="rtl-wrapper">
    <div class="logo-wrapper">
        <img src="data:image/png;base64,{logo_b64}">
    </div>
    <h2 style="text-align: center; color: #e8dfc8;">تصنيف وتبسيط النصوص العربية</h2>
    <div class="gold-divider"></div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------
# Main Logic Wrapped in RTL Div
# -----------------------------------------
st.markdown('<div class="rtl-wrapper">', unsafe_allow_html=True)

text = st.text_area("أدخل النص العربي", height=200, placeholder="اكتب أو الصق النص هنا...")

if 'done' not in st.session_state:
    st.session_state.done = False

if st.button("تحليل", use_container_width=True, type="primary"):
    if text.strip():
        with st.spinner("جاري التحليل..."):
            cleaned = normalize_ar(text)
            level, conf = classify(cleaned)
            st.session_state.done  = True
            st.session_state.level = level
            st.session_state.conf  = conf
            st.session_state.text  = text
    else:
        st.warning("⚠️ الرجاء إدخال نص")

if st.session_state.done:
    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)
    
    level = st.session_state.level
    conf  = st.session_state.conf
    level_colors = {1: "🟢", 2: "🟢", 3: "🟡", 4: "🟡", 5: "🔴", 6: "🔴"}
    level_names  = {1: "سهل جداً", 2: "سهل", 3: "متوسط", 4: "صعب قليلاً", 5: "صعب", 6: "صعب جداً"}

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="المستوى", value=f"{level_colors.get(level,'⚪')} {level}")
    with col2:
        st.metric(label="الوصف", value=level_names.get(level, "غير معروف"))

    st.progress(int(conf * 100))
    st.markdown(f"**نسبة الثقة:** {conf:.2%}")

    if level >= 4:
        st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)
        st.info("💡 هذا النص صعب القراءة. يمكنك تبسيطه بالضغط على الزر أدناه.")

        if st.button("تبسيط", use_container_width=True, type="secondary"):
            if simplifier_model:
                with st.spinner("جاري التبسيط..."):
                    cleaned = normalize_ar(st.session_state.text)
                    inputs  = simplifier_tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=512)
                    with torch.no_grad():
                        outputs = simplifier_model.generate(**inputs, max_length=512, num_beams=4)
                    simplified = simplifier_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    st.markdown("<p style='color:#D4AF37; font-size:1.2rem;'>النص المبسط</p>", unsafe_allow_html=True)
                    st.markdown(f"<div class='simplified-box'>{simplified}</div>", unsafe_allow_html=True)
            else:
                st.warning("⚠️ نموذج التبسيط غير متوفر حالياً.")

st.markdown('</div>', unsafe_allow_html=True) # End RTL Wrapper

st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)
st.caption("© 2025 — مشروع بَيِّنْ")
