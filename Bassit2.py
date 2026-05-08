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
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)
import torch
import numpy as np

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="بَيِّنْ وَ بَسِّطْ",
    page_icon="📖",
    layout="centered"
)

# =========================================================
# IMAGE HELPER
# =========================================================
def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

# =========================================================
# ARABIC NORMALIZATION
# =========================================================
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

# =========================================================
# LOAD MODELS
# =========================================================
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
        tokenizer = AutoTokenizer.from_pretrained(
            "SarahAlhalees/bassit-simplifier"
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            "SarahAlhalees/bassit-simplifier"
        )

        return tokenizer, model

    except:
        return None, None

classifier_tokenizer, classifier_model = load_classifier()
simplifier_tokenizer, simplifier_model = load_simplifier()

# =========================================================
# CLASSIFICATION
# =========================================================
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

# =========================================================
# LEVEL TOKENS
# =========================================================
LEVEL_TOKEN = {
    "mild": "[L3]",
    "medium": "[L2]",
    "strong": "[L1]",
}

LEVEL_LABELS = {
    "mild": "تبسيط خفيف",
    "medium": "تبسيط متوسط",
    "strong": "تبسيط قوي",
}

LEVEL_ICONS = {
    "mild": "✦",
    "medium": "✦✦",
    "strong": "✦✦✦",
}

# =========================================================
# SIMPLIFICATION
# =========================================================
def simplify(text, level_key):

    prefix = LEVEL_TOKEN[level_key]

    cleaned = normalize_ar(text)

    source = f"{prefix} {cleaned}"

    inputs = simplifier_tokenizer(
        source,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    outputs = simplifier_model.generate(
        **inputs,
        max_length=512,
        num_beams=4,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        min_length=10,
    )

    return simplifier_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

# =========================================================
# VALIDATION
# =========================================================
def is_valid_arabic(text):

    check_text = re.sub(r"[\s\d\W_]+", "", text)

    if not check_text:
        return False, "الرجاء إدخال نص (ليس أرقاماً فقط)"

    if not re.search(r"[\u0600-\u06FF]", text):
        return False, "الرجاء إدخال نص باللغة العربية فقط"

    return True, ""

# =========================================================
# SESSION STATE
# =========================================================
for key in (
    "done",
    "level",
    "conf",
    "text",
    "simplified_results"
):

    if key not in st.session_state:

        st.session_state[key] = (
            False if key == "done"
            else {} if key == "simplified_results"
            else None
        )

# =========================================================
# LOAD IMAGES
# =========================================================
logo_b64 = get_image_base64("logo4.png")
bg_b64 = get_image_base64("jamal.jpg")

# =========================================================
# STYLING
# =========================================================
st.markdown(f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Cairo:wght@300;400;700&display=swap');

/* =====================================================
   BACKGROUND
===================================================== */

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
    background: linear-gradient(
        180deg,
        rgba(10, 25, 40, 0.78) 0%,
        rgba(20, 35, 45, 0.92) 100%
    );
    z-index: 0;
}}

.block-container {{
    position: relative;
    z-index: 1;
    max-width: 820px;
    padding-top: 2rem;
}}

/* =====================================================
   TYPOGRAPHY
===================================================== */

h1, h2, h3, p, span, label {{
    font-family: 'Cairo', sans-serif !important;
    text-align: right !important;
    direction: rtl !important;
    color: #F5EEDC !important;
}}

/* =====================================================
   LOGO
===================================================== */

.logo-wrapper {{
    display: flex;
    justify-content: center;
    margin-top: 3rem;
    margin-bottom: 1rem;
}}

.logo-wrapper img {{
    height: 180px;
    filter: drop-shadow(0 0 15px rgba(197, 160, 89, 0.35));
}}

/* =====================================================
   SUBTITLE
===================================================== */

.app-subtitle {{
    text-align: center !important;
    font-family: 'Amiri', serif !important;
    font-size: 1.35rem;
    color: #FFFFFF !important;
    margin-bottom: 2rem;
}}

/* =====================================================
   DIVIDER
===================================================== */

.gold-divider {{
    height: 1px;
    background: linear-gradient(
        90deg,
        transparent,
        #C5A059,
        transparent
    );
    width: 50%;
    margin: 2rem auto;
    opacity: 0.6;
}}

/* =====================================================
   TEXT AREA
===================================================== */

textarea {{
    direction: rtl !important;
    text-align: right !important;

    background: rgba(15, 30, 45, 0.80) !important;

    color: #F5EEDC !important;

    border: 1px solid rgba(197, 160, 89, 0.4) !important;

    border-radius: 15px !important;

    padding: 15px !important;

    font-size: 1.1rem !important;

    caret-color: #F5EEDC !important;
}}

textarea::placeholder {{
    color: rgba(245, 238, 220, 0.45) !important;
}}

/* =====================================================
   MAIN BUTTON
===================================================== */

.stButton {{
    display: flex;
    justify-content: center;
}}

.stButton > button {{
    border-radius: 14px !important;
    transition: all 0.25s ease !important;
    font-family: 'Cairo', sans-serif !important;
}}

.stButton > button[kind="primary"] {{

    width: 220px !important;

    height: 3.7rem !important;

    font-size: 1.2rem !important;

    font-weight: 700 !important;

    border: none !important;

    background: linear-gradient(
        135deg,
        #C5A059 0%,
        #8E733E 100%
    ) !important;

    color: #0E1E2B !important;

    box-shadow: 0 4px 18px rgba(197,160,89,0.35) !important;
}}

.stButton > button[kind="primary"]:hover {{

    transform: translateY(-3px);

    box-shadow: 0 8px 28px rgba(197,160,89,0.45) !important;
}}

/* =====================================================
   LEVEL LABEL
===================================================== */

.simplify-label {{
    text-align: center !important;
    font-family: 'Cairo', sans-serif !important;
    font-size: 1.05rem;
    color: rgba(197,160,89,0.9) !important;
    margin-bottom: 1rem;
}}

/* =====================================================
   MOBILE CENTERED BUTTONS
===================================================== */

.simplify-buttons-wrapper {{

    display: flex;
    flex-direction: column;

    align-items: center;
    justify-content: center;

    gap: 14px;

    width: 100%;

    margin-top: 1rem;
    margin-bottom: 1rem;
}}

.simplify-buttons-wrapper .stButton {{

    width: 100%;

    display: flex;

    justify-content: center;
}}

.simplify-buttons-wrapper .stButton > button {{

    width: 260px !important;

    max-width: 90vw !important;

    height: 4rem !important;

    border-radius: 18px !important;

    background: rgba(12, 24, 36, 0.92) !important;

    color: #F5EEDC !important;

    border: 1.5px solid rgba(197, 160, 89, 0.65) !important;

    font-size: 1.08rem !important;

    font-family: 'Cairo', sans-serif !important;

    font-weight: 700 !important;

    box-shadow: 0 4px 15px rgba(0,0,0,0.35) !important;
}}

.simplify-buttons-wrapper .stButton > button:hover {{

    background: rgba(197,160,89,0.15) !important;

    border-color: #C5A059 !important;

    color: #C5A059 !important;

    transform: translateY(-2px);

    box-shadow: 0 8px 22px rgba(197,160,89,0.25) !important;
}}

@media (max-width: 480px) {{

    .simplify-buttons-wrapper .stButton > button {{

        width: 90vw !important;

        max-width: 320px !important;

        font-size: 1rem !important;

        height: 3.8rem !important;
    }}
}}

/* =====================================================
   RESULT PILL
===================================================== */

.stat-pill {{

    display: inline-block;

    padding: 10px 18px;

    border-radius: 12px;

    background: rgba(8,18,30,0.82);

    border: 1px solid rgba(197,160,89,0.35);

    font-size: 1.1rem;

    color: #F5EEDC;

    text-align: center;

    width: 100%;
}}

.stat-pill .gold {{
    color: #C5A059;
    font-weight: 700;
    font-size: 1.35rem;
}}

/* =====================================================
   RESULT BOX
===================================================== */

.simplified-box {{

    background: rgba(8,18,28,0.88);

    padding: 28px 30px;

    border-radius: 15px;

    color: #F5EEDC !important;

    border-right: 5px solid #C5A059;

    margin-top: 20px;

    line-height: 2;

    font-size: 1.12rem;

    direction: rtl;

    text-align: right;
}}

.box-label {{

    color: #C5A059 !important;

    font-weight: 700;

    display: block;

    margin-bottom: 10px;
}}

.box-text {{
    color: #F0E8D5 !important;
}}

</style>

<div class="logo-wrapper">
    <img src="data:image/png;base64,{logo_b64}">
</div>

<div class="app-subtitle">
نظام ذكي لتصنيف مستوى مقروئية النصوص العربية وتبسيطها
</div>

<div class="gold-divider"></div>

""", unsafe_allow_html=True)

# =========================================================
# TEXT INPUT
# =========================================================
text = st.text_area(
    "أدخل النص المراد تصنيفه:",
    height=220,
    placeholder="اكتب أو الصق النص هنا..."
)

# =========================================================
# MAIN BUTTON
# =========================================================
col1, col2, col3 = st.columns([1,1,1])

with col2:

    classify_btn = st.button(
        "بَيِّنْ",
        type="primary"
    )

if classify_btn:

    if text.strip():

        is_valid, error_msg = is_valid_arabic(text)

        if not is_valid:

            st.error(error_msg)

            st.session_state.done = False

        else:

            with st.spinner("يتم الآن فحص لغة النص..."):

                cleaned = normalize_ar(text)

                level, conf = classify(cleaned)

                st.session_state.done = True
                st.session_state.level = level
                st.session_state.conf = conf
                st.session_state.text = text

                st.session_state.simplified_results = {}

    else:
        st.error("الرجاء تزويدنا بنص للبدء")

# =========================================================
# RESULTS
# =========================================================
if st.session_state.done:

    st.markdown(
        "<div class='gold-divider'></div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class='stat-pill'>
            مستوى الصعوبة:
            <span class='gold'>
                {st.session_state.level}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # SIMPLIFICATION BUTTONS
    # =====================================================
    if st.session_state.level >= 4:

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class='simplify-label'>
                اختر درجة التبسيط المطلوبة:
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="simplify-buttons-wrapper">',
            unsafe_allow_html=True
        )

        btn_mild = st.button(
            "✦ تبسيط خفيف",
            key="btn_mild"
        )

        btn_medium = st.button(
            "✦✦ تبسيط متوسط",
            key="btn_medium"
        )

        btn_strong = st.button(
            "✦✦✦ تبسيط قوي",
            key="btn_strong"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        # =================================================
        # BUTTON ACTIONS
        # =================================================
        if btn_mild:

            if simplifier_model:

                with st.spinner("جاري التبسيط الخفيف..."):

                    st.session_state.simplified_results["mild"] = simplify(
                        st.session_state.text,
                        "mild"
                    )

            else:
                st.error("خدمة التبسيط غير متاحة حالياً")

        if btn_medium:

            if simplifier_model:

                with st.spinner("جاري التبسيط المتوسط..."):

                    st.session_state.simplified_results["medium"] = simplify(
                        st.session_state.text,
                        "medium"
                    )

            else:
                st.error("خدمة التبسيط غير متاحة حالياً")

        if btn_strong:

            if simplifier_model:

                with st.spinner("جاري التبسيط القوي..."):

                    st.session_state.simplified_results["strong"] = simplify(
                        st.session_state.text,
                        "strong"
                    )

            else:
                st.error("خدمة التبسيط غير متاحة حالياً")

        # =================================================
        # SHOW RESULTS
        # =================================================
        for level_key in ("mild", "medium", "strong"):

            result = st.session_state.simplified_results.get(level_key)

            if result:

                icon = LEVEL_ICONS[level_key]
                label = LEVEL_LABELS[level_key]

                st.markdown(
                    f"""
                    <div class='simplified-box'>

                        <span class='box-label'>
                            {icon} النتيجة — {label}
                        </span>

                        <div class='box-text'>
                            {result}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div style="
    text-align:center;
    color:#C5A059;
    margin-top:60px;
    font-size:0.85rem;
    opacity:0.65;
    font-family:'Cairo';
">
© 2026 — مشروع بَيِّنْ وَ بَسِّطْ
</div>
""", unsafe_allow_html=True)
