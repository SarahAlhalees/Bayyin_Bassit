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
# Streamlit Page Config
# -----------------------------------------
st.set_page_config(
    page_title="بَيِّنْ - تصنيف وتبسيط النصوص العربية",
    page_icon="📖",
    layout="centered"
)

# -----------------------------------------
# Arabic Text Normalization
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


# Load at startup
classifier_tokenizer, classifier_model = load_classifier()
simplifier_tokenizer,  simplifier_model  = load_simplifier()

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
    probs      = torch.softmax(logits, dim=-1).squeeze().numpy()
    prediction = int(np.argmax(probs)) + 1   # labels are 1-indexed (1-6)
    confidence = float(probs[prediction - 1])
    return prediction, confidence

# -----------------------------------------
# Theme & UI
# -----------------------------------------
logo_b64 = get_image_base64("logo2.png")
bg_b64   = get_image_base64("jamal.jpg")

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Cairo:wght@400;600;700&display=swap');

    /* Hide Streamlit chrome */
    #MainMenu, footer, header {{visibility: hidden;}}

    /* Background */
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bg_b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        font-family: 'Cairo', sans-serif;
    }}

    /* Navy overlay */
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: linear-gradient(
            160deg,
            rgba(10, 25, 60, 0.82) 0%,
            rgba(10, 25, 60, 0.65) 50%,
            rgba(10, 25, 60, 0.80) 100%
        );
        z-index: 0;
    }}

    .block-container {{
        position: relative;
        z-index: 1;
        padding-top: 1rem !important;
        max-width: 780px;
    }}

    /* Logo */
    .logo-wrapper {{
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1.5rem auto 0.5rem auto;
    }}
    .logo-wrapper img {{
        height: 140px;
        width: auto;
        object-fit: contain;
        filter: drop-shadow(0 4px 16px rgba(212,175,55,0.45));
    }}

    /* Title */
    .app-title {{
        font-family: 'Amiri', serif;
        font-size: 3.2rem;
        font-weight: 700;
        text-align: center;
        color: #D4AF37;
        text-shadow: 0 2px 12px rgba(0,0,0,0.5);
        margin: 0.2rem 0 0.1rem 0;
        direction: rtl;
        letter-spacing: 2px;
    }}
    .app-subtitle {{
        font-family: 'Cairo', sans-serif;
        font-size: 1.15rem;
        font-weight: 400;
        text-align: center;
        color: #e8dfc8;
        margin-bottom: 1rem;
        direction: rtl;
        opacity: 0.9;
    }}

    /* Gold divider */
    .gold-divider {{
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #D4AF37, transparent);
        margin: 0.8rem auto 1.4rem auto;
        width: 60%;
    }}

    /* Text area */
    textarea {{
        direction: rtl !important;
        text-align: right !important;
        font-size: 16px !important;
        font-family: 'Cairo', sans-serif !important;
        background-color: rgba(10, 25, 60, 0.75) !important;
        color: #f0e6c8 !important;
        border: 1.5px solid #D4AF37 !important;
        border-radius: 10px !important;
    }}
    textarea::placeholder {{ color: #a89060 !important; }}

    /* Labels */
    label, .stTextArea label {{
        color: #D4AF37 !important;
        font-family: 'Cairo', sans-serif !important;
        font-size: 1rem !important;
        direction: rtl;
    }}

    /* Primary button */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #D4AF37 0%, #b8922a 100%) !important;
        color: #0a1940 !important;
        font-family: 'Amiri', serif !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        box-shadow: 0 4px 18px rgba(212,175,55,0.35) !important;
        transition: all 0.2s ease !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: linear-gradient(135deg, #e8c84a 0%, #D4AF37 100%) !important;
        box-shadow: 0 6px 24px rgba(212,175,55,0.55) !important;
        transform: translateY(-1px) !important;
    }}

    /* Secondary button */
    .stButton > button[kind="secondary"] {{
        background: rgba(10, 25, 60, 0.8) !important;
        color: #D4AF37 !important;
        font-family: 'Amiri', serif !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        border: 1.5px solid #D4AF37 !important;
        border-radius: 10px !important;
        padding: 0.5rem 2rem !important;
        transition: all 0.2s ease !important;
    }}
    .stButton > button[kind="secondary"]:hover {{
        background: rgba(212,175,55,0.15) !important;
        box-shadow: 0 4px 16px rgba(212,175,55,0.3) !important;
    }}

    /* Metric cards */
    [data-testid="metric-container"] {{
        background: rgba(10, 25, 60, 0.75) !important;
        border: 1px solid #D4AF37 !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        text-align: center !important;
    }}
    [data-testid="metric-container"] label {{
        color: #D4AF37 !important;
        font-family: 'Cairo', sans-serif !important;
    }}
    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: #f0e6c8 !important;
        font-family: 'Amiri', serif !important;
        font-size: 1.6rem !important;
    }}

    /* Progress bar */
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, #D4AF37, #b8922a) !important;
        border-radius: 8px !important;
    }}
    .stProgress > div > div {{
        background: rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
    }}

    /* Alert boxes */
    .stAlert {{
        background: rgba(10, 25, 60, 0.8) !important;
        border-color: #D4AF37 !important;
        color: #f0e6c8 !important;
        border-radius: 10px !important;
    }}

    /* Subheaders */
    h2, h3 {{
        color: #D4AF37 !important;
        font-family: 'Amiri', serif !important;
        direction: rtl;
        text-align: right;
    }}

    /* Body text */
    p, .stMarkdown p {{
        color: #e8dfc8 !important;
        font-family: 'Cairo', sans-serif !important;
        direction: rtl;
        text-align: right;
    }}

    /* Simplified result box */
    .simplified-box {{
        background: rgba(10, 25, 60, 0.85);
        padding: 22px 26px;
        border-radius: 12px;
        direction: rtl;
        text-align: right;
        color: #f0e6c8;
        border-right: 4px solid #D4AF37;
        border-top: 1px solid rgba(212,175,55,0.3);
        margin-top: 20px;
        font-family: 'Cairo', sans-serif;
        font-size: 1.05rem;
        line-height: 1.9;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }}

    /* Caption */
    .stCaption, small {{
        color: #a89060 !important;
        text-align: center;
        display: block;
    }}

    /* Spinner */
    .stSpinner > div {{
        border-top-color: #D4AF37 !important;
    }}
    </style>

    <!-- Logo -->
    <div class="logo-wrapper">
        <img src="data:image/jpeg;base64,{logo_b64}" alt="Bayyin Logo"/>
    </div>

    <!-- Title -->
    <div class="app-title">بَيِّنْ</div>
    <div class="app-subtitle">مصنِّف وتبسيط النصوص العربية</div>
    <div class="gold-divider"></div>
""", unsafe_allow_html=True)

# -----------------------------------------
# Input
# -----------------------------------------
text = st.text_area(
    label="أدخل النص العربي",
    height=200,
    placeholder="اكتب أو الصق النص هنا...",
    key="arabic_input"
)

# Session state
if 'classification_done' not in st.session_state:
    st.session_state.classification_done = False
if 'readability_level' not in st.session_state:
    st.session_state.readability_level = 0
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.0
if 'original_text' not in st.session_state:
    st.session_state.original_text = ""

# -----------------------------------------
# بَيِّنْ Button
# -----------------------------------------
if st.button("بَيِّنْ", use_container_width=True, type="primary"):
    if not text.strip():
        st.warning("⚠️ الرجاء إدخال نص.")
    elif classifier_model is None:
        st.error("⚠️ لم يتم تحميل النموذج.")
    else:
        with st.spinner("جاري التحليل..."):
            cleaned = normalize_ar(text)
            prediction, confidence = classify(cleaned)

            st.session_state.classification_done = True
            st.session_state.readability_level   = prediction
            st.session_state.confidence          = confidence
            st.session_state.original_text       = text

# -----------------------------------------
# Display Results
# -----------------------------------------
if st.session_state.classification_done:
    st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)
    st.subheader("📊 نتيجة التصنيف")

    level = st.session_state.readability_level
    level_colors = {1: "🟢", 2: "🟢", 3: "🟡", 4: "🟡", 5: "🔴", 6: "🔴"}
    level_names  = {
        1: "سهل جداً",
        2: "سهل",
        3: "متوسط",
        4: "صعب قليلاً",
        5: "صعب",
        6: "صعب جداً"
    }

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="المستوى", value=f"{level_colors.get(level, '⚪')} {level}")
    with col2:
        st.metric(label="الوصف", value=level_names.get(level, "غير معروف"))

    st.progress(int(st.session_state.confidence * 100))
    st.write(f"**نسبة الثقة:** {st.session_state.confidence:.2%}")

    # بَسِّطْ Button (levels 4-6 only)
    if level >= 4:
        st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)
        st.info("💡 هذا النص صعب القراءة. يمكنك تبسيطه بالضغط على الزر أدناه.")

        if st.button("بَسِّطْ", use_container_width=True, type="secondary"):
            if simplifier_model and simplifier_tokenizer:
                with st.spinner("جاري التبسيط..."):
                    cleaned = normalize_ar(st.session_state.original_text)
                    inputs  = simplifier_tokenizer(
                        cleaned,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=512
                    )
                    with torch.no_grad():
                        outputs = simplifier_model.generate(
                            **inputs,
                            max_length=512,
                            num_beams=4,
                            length_penalty=1.0,
                            early_stopping=True,
                            no_repeat_ngram_size=3
                        )
                    simplified_text = simplifier_tokenizer.decode(
                        outputs[0], skip_special_tokens=True
                    )
                    st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)
                    st.subheader("النص المبسط")
                    st.markdown(
                        f'<div class="simplified-box">{simplified_text}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.warning("⚠️نموذج التبسيط غير متوفر حالياً.")

st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)
st.caption("© 2025 — مشروع بَيِّنْ")
