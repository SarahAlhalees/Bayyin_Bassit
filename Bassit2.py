import os
import warnings
import logging
import sys
import types

# Suppress transformers __path__ scanning noise and deprecation warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModel
import torch
import torch.nn as nn
import numpy as np
import re
import joblib
from huggingface_hub import hf_hub_download

import base64

def get_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# -----------------------------------------
# BiLSTM Model Class Definition (MUST BE BEFORE LOADING)
# -----------------------------------------
class BiLSTMWithMeta(nn.Module):
    def __init__(self, input_dim, categorical_cardinalities, num_numeric,
                 lstm_hidden=256, meta_proj_dim=128, num_classes=6, dropout=0.3,
                 use_bert=False, bert_model_name=None):
        super().__init__()
        self.use_bert = use_bert
        if use_bert and bert_model_name:
            self.bert = AutoModel.from_pretrained(bert_model_name)
            input_dim = self.bert.config.hidden_size
        else:
            self.bert = None

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden,
                            num_layers=1, batch_first=True, bidirectional=True)

        self.cat_names = list(categorical_cardinalities.keys())
        self.cat_embeddings = nn.ModuleDict()
        total_cat_emb_dim = 0
        for name, card in categorical_cardinalities.items():
            emb_dim = min(50, max(4, int(card**0.5)))
            self.cat_embeddings[name] = nn.Embedding(card, emb_dim)
            total_cat_emb_dim += emb_dim

        self.meta_proj = nn.Sequential(
            nn.Linear(total_cat_emb_dim + num_numeric, meta_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2 + meta_proj_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, numeric_meta, categorical_meta, attention_mask=None):
        if self.use_bert and self.bert is not None:
            bert_out = self.bert(input_ids=x, attention_mask=attention_mask)
            x = bert_out.last_hidden_state
        else:
            if len(x.shape) == 2:
                x = x.unsqueeze(1)

        lstm_out, _ = self.lstm(x)
        pooled = lstm_out.mean(dim=1) if self.use_bert else lstm_out.squeeze(1)

        cat_embs = [self.cat_embeddings[name](categorical_meta[:, i])
                    for i, name in enumerate(self.cat_names)]
        cat_concat = torch.cat(cat_embs, dim=1) if cat_embs else \
                     torch.zeros(numeric_meta.size(0), 0, device=numeric_meta.device)

        meta_concat = torch.cat([numeric_meta, cat_concat], dim=1)
        meta_vec = self.meta_proj(meta_concat)

        fused = torch.cat([pooled, meta_vec], dim=1)
        fused = self.dropout(fused)
        return self.classifier(fused)


class BiLSTMWrapper:
    def __init__(self, model, cat_cardinalities, num_numeric, num_classes=6, device='cpu'):
        self.device = device
        self.num_classes = num_classes
        self.cat_cardinalities = cat_cardinalities
        self.num_numeric = num_numeric
        self.model = model
        self.default_numeric = np.zeros(num_numeric, dtype=np.float32)
        self.default_categorical = np.zeros(len(cat_cardinalities), dtype=np.int64)

    def predict(self, X):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
            if len(X.shape) == 1:
                X = X.unsqueeze(0)

            batch_size = X.shape[0]
            numeric = torch.tensor(np.tile(self.default_numeric, (batch_size, 1)), dtype=torch.float32).to(self.device)
            categorical = torch.tensor(np.tile(self.default_categorical, (batch_size, 1)), dtype=torch.long).to(self.device)

            logits = self.model(X.to(self.device), numeric, categorical)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            return predictions + 1

    def predict_proba(self, X):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
            if len(X.shape) == 1:
                X = X.unsqueeze(0)

            batch_size = X.shape[0]
            numeric = torch.tensor(np.tile(self.default_numeric, (batch_size, 1)), dtype=torch.float32).to(self.device)
            categorical = torch.tensor(np.tile(self.default_categorical, (batch_size, 1)), dtype=torch.long).to(self.device)

            logits = self.model(X.to(self.device), numeric, categorical)
            return torch.softmax(logits, dim=1).cpu().numpy()


# Register classes for joblib unpickling
current_module = sys.modules[__name__]
current_module.BiLSTMWrapper = BiLSTMWrapper
current_module.BiLSTMWithMeta = BiLSTMWithMeta

if '__main__' not in sys.modules:
    sys.modules['__main__'] = types.ModuleType('__main__')
sys.modules['__main__'].BiLSTMWrapper = BiLSTMWrapper
sys.modules['__main__'].BiLSTMWithMeta = BiLSTMWithMeta

if 'main' not in sys.modules:
    sys.modules['main'] = current_module
else:
    sys.modules['main'].BiLSTMWrapper = BiLSTMWrapper
    sys.modules['main'].BiLSTMWithMeta = BiLSTMWithMeta

# -----------------------------------------
# Streamlit Page Settings
# -----------------------------------------
st.set_page_config(
    page_title="بَيِّنْ - تصنيف وتبسيط النصوص العربية",
    page_icon="📖",
    layout="centered"
)

# -----------------------------------------
# Arabic text normalization
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
def load_base_models():
    models = {}

    # --- AraBERT v2 (fine-tuned classifier) ---
    try:
        try:
            models['arabert_tokenizer'] = AutoTokenizer.from_pretrained(
                "SarahAlhalees/AraBERTv2_RefinedBayyin",
                use_fast=False
            )
            models['arabert_model'] = AutoModelForSequenceClassification.from_pretrained(
                "SarahAlhalees/AraBERTv2_RefinedBayyin"
            )
        except Exception:
            models['arabert_tokenizer'] = AutoTokenizer.from_pretrained(
                "SarahAlhalees/Arabertv2_D3Tok",
                subfolder="Arabertv2_D3Tok",
                use_fast=False
            )
            models['arabert_model'] = AutoModelForSequenceClassification.from_pretrained(
                "SarahAlhalees/Arabertv2_D3Tok",
                subfolder="Arabertv2_D3Tok"
            )
        models['arabert_model'].eval()
    except Exception as e:
        st.warning(f"خطأ في تحميل AraBERT: {str(e)}")
        models['arabert_tokenizer'] = None
        models['arabert_model'] = None

    # --- CamelBERT-MSA (fine-tuned classifier) ---
    try:
        models['camelbert_tokenizer'] = AutoTokenizer.from_pretrained(
            "SarahAlhalees/CamelBERTmsa_RefinedBayyin",
            use_fast=False
        )
        models['camelbert_model'] = AutoModelForSequenceClassification.from_pretrained(
            "SarahAlhalees/CamelBERTmsa_RefinedBayyin"
        )
        models['camelbert_model'].eval()
    except Exception as e:
        st.warning(f"خطأ في تحميل CamelBERT: {str(e)}")
        models['camelbert_tokenizer'] = None
        models['camelbert_model'] = None

    # --- BiLSTM + CamelBERT-MSA backbone (end-to-end, BERT is inside the model) ---
    try:
        bilstm_path = hf_hub_download(
            repo_id="SarahAlhalees/BiLSTM_RefinedBayyin",
            filename="best_bilstm_camelbert.joblib"
        )
        meta_enc_path = hf_hub_download(
            repo_id="SarahAlhalees/BiLSTM_RefinedBayyin",
            filename="meta_encoders.joblib"
        )

        bilstm_raw  = joblib.load(bilstm_path)
        config      = bilstm_raw['config']
        cat_card    = bilstm_raw['cat_card']
        model_state = bilstm_raw['model_state']

        # --- Infer dims from state_dict ---
        total_cat_emb = sum(
            min(50, max(4, int(card**0.5)))
            for card in (cat_card.values() if isinstance(cat_card, dict) else [])
        )

        meta_proj_weight = model_state.get('meta_proj.0.weight')
        if meta_proj_weight is not None:
            meta_proj_in = meta_proj_weight.shape[1]
            num_numeric  = meta_proj_in - total_cat_emb
        else:
            num_numeric = config.get('num_numeric', 0) if isinstance(config, dict) else 0

        if isinstance(config, dict):
            lstm_hidden   = config.get('lstm_hidden',   256)
            meta_proj_dim = config.get('meta_proj_dim', 128)
            num_classes   = config.get('num_classes',   6)
            dropout       = config.get('dropout',       0.3)
            bert_model_name = config.get('bert_model_name', 'CAMeL-Lab/bert-base-arabic-camelbert-msa')
        else:
            lstm_hidden   = getattr(config, 'lstm_hidden',   256)
            meta_proj_dim = getattr(config, 'meta_proj_dim', 128)
            num_classes   = getattr(config, 'num_classes',   6)
            dropout       = getattr(config, 'dropout',       0.3)
            bert_model_name = getattr(config, 'bert_model_name', 'CAMeL-Lab/bert-base-arabic-camelbert-msa')

        # Override lstm_hidden from state_dict if available
        lstm_weight = model_state.get('lstm.weight_hh_l0')
        if lstm_weight is not None:
            lstm_hidden = lstm_weight.shape[1]

        # input_dim is determined by BERT hidden size when use_bert=True,
        # so we pass a placeholder — it gets overridden inside __init__
        input_dim = 768

        # Reconstruct BiLSTMWithMeta with use_bert=True (BERT lives inside the model)
        bilstm_nn = BiLSTMWithMeta(
            input_dim                 = input_dim,
            categorical_cardinalities = cat_card if isinstance(cat_card, dict) else {},
            num_numeric               = num_numeric,
            lstm_hidden               = lstm_hidden,
            meta_proj_dim             = meta_proj_dim,
            num_classes               = num_classes,
            dropout                   = dropout,
            use_bert                  = True,
            bert_model_name           = bert_model_name,
        )
        bilstm_nn.load_state_dict(model_state, strict=False)
        bilstm_nn.eval()

        models['bilstm_model'] = BiLSTMWrapper(
            model             = bilstm_nn,
            cat_cardinalities = cat_card if isinstance(cat_card, dict) else {},
            num_numeric       = num_numeric,
            num_classes       = num_classes,
        )
        models['bilstm_raw'] = bilstm_raw

        # CamelBERT-MSA tokenizer (for tokenizing input to the BiLSTM model)
        models['bilstm_tokenizer'] = AutoTokenizer.from_pretrained(
            bert_model_name,
            use_fast=False
        )

        # Load meta_encoders (scaler is for numeric metadata features, not used in inference)
        meta_raw = joblib.load(meta_enc_path)
        models['meta_encoders']      = meta_raw.get('meta_scaler')
        models['label_encoders']     = meta_raw.get('label_encoders')
        models['meta_encoders_dict'] = meta_raw

        # NOTE: No external bilstm_bert needed — BERT is inside BiLSTMWithMeta

    except Exception as e:
        st.warning(f"خطأ في تحميل BiLSTM: {str(e)}")
        models['bilstm_model']        = None
        models['bilstm_tokenizer']    = None
        models['meta_encoders']       = None
        models['meta_encoders_dict']  = None

    return models


@st.cache_resource
def load_meta_learner():
    """Load stacking meta-learner (SVM trained on Stage-2 features)"""
    try:
        meta_path = hf_hub_download(
            repo_id="SarahAlhalees/ensemble",
            filename="meta_svm_tuned.joblib"
        )
        return joblib.load(meta_path)
    except Exception as e:
        st.error(f"خطأ في تحميل Meta-Learner: {str(e)}")
        return None


@st.cache_resource
def load_simplification_model():
    """Load AraBART text simplification model"""
    try:
        repo_id = "SarahAlhalees/bassit-simplifier"
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
        return tokenizer, model
    except Exception as e:
        st.warning(f"نموذج التبسيط غير متوفر: {str(e)}")
        return None, None


# Load all models at startup
_load_error = None
try:
    base_models = load_base_models()
except Exception as _e:
    _load_error = f"load_base_models failed:\n{type(_e).__name__}: {_e}"
    import traceback as _tb
    _load_error += "\n\n" + _tb.format_exc()
    base_models = {}

try:
    meta_learner = load_meta_learner()
except Exception as _e:
    if _load_error is None:
        _load_error = ""
    import traceback as _tb
    _load_error += f"\nload_meta_learner failed: {type(_e).__name__}: {_e}\n{_tb.format_exc()}"
    meta_learner = None

try:
    simplifier_tokenizer, simplifier_model = load_simplification_model()
except Exception as _e:
    simplifier_tokenizer, simplifier_model = None, None

if _load_error:
    st.error("🚨 خطأ أثناء تحميل النماذج — تفاصيل للمطور:")
    st.code(_load_error, language="python")
    st.stop()

# --- Debug expander ---
with st.expander("🔧 تشخيص تحميل النماذج", expanded=False):
    st.write(f"**arabert loaded:** `{base_models.get('arabert_model') is not None}`")
    st.write(f"**camelbert loaded:** `{base_models.get('camelbert_model') is not None}`")
    st.write(f"**bilstm_model type:** `{type(base_models.get('bilstm_model'))}`")
    st.write(f"**meta_scaler type:** `{type(base_models.get('meta_encoders'))}`")
    st.write(f"**label_encoders:** `{type(base_models.get('label_encoders'))}`")
    st.write(f"**meta_learner loaded:** `{meta_learner is not None}`")

    bilstm_wrapper = base_models.get('bilstm_model')
    if bilstm_wrapper:
        st.write(f"**num_numeric:** `{bilstm_wrapper.num_numeric}`")
        st.write(f"**cat_cardinalities count:** `{len(bilstm_wrapper.cat_cardinalities)}`")
        st.write(f"**use_bert:** `{bilstm_wrapper.model.use_bert}`")

    raw = base_models.get('bilstm_raw') or {}
    if raw:
        cfg = raw.get('config', {})
        st.write(f"**config:** `{cfg}`")
        st.write(f"**cat_card keys:** `{list(raw.get('cat_card', {}).keys())}`")

    scaler = base_models.get('meta_encoders')
    if scaler and hasattr(scaler, 'n_features_in_'):
        st.write(f"**meta_scaler.n_features_in_:** `{scaler.n_features_in_}`")

# -----------------------------------------
# Feature Extraction
# -----------------------------------------
def get_softmax_probs_from_classifier(text, tokenizer, clf_model, max_length=256):
    """
    Return 6-class softmax probabilities from a fine-tuned classifier model.
    Shape: (1, 6)
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    )
    with torch.no_grad():
        logits = clf_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).numpy()
    return probs


def get_bilstm_probs(text, models, max_length=256):
    """
    Get 6-class softmax probabilities from the BiLSTM+CamelBERT-MSA model.

    The model has BERT embedded inside it (use_bert=True), so we:
      1. Tokenize with the CamelBERT-MSA tokenizer → token IDs
      2. Pass token IDs directly into bilstm_nn.forward()
      3. Use zeroed numeric/categorical metadata (not available at inference time)

    The meta_scaler (4 features) is for numeric metadata columns used during
    training only — it is NOT applied to BERT output here.

    Shape returned: (1, 6)
    """
    tokenizer = models['bilstm_tokenizer']
    bilstm    = models['bilstm_model']      # BiLSTMWrapper

    # Tokenize → token IDs
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    )

    input_ids      = inputs['input_ids']       # (1, seq_len)
    attention_mask = inputs['attention_mask']  # (1, seq_len)

    # Call the model directly (bypassing BiLSTMWrapper.predict_proba)
    # because we need to pass attention_mask for the internal BERT
    bilstm.model.eval()
    bilstm.model.to(bilstm.device)

    with torch.no_grad():
        batch_size  = input_ids.shape[0]
        numeric     = torch.zeros(batch_size, bilstm.num_numeric, dtype=torch.float32).to(bilstm.device)
        categorical = torch.zeros(batch_size, len(bilstm.cat_cardinalities), dtype=torch.long).to(bilstm.device)

        logits = bilstm.model(
            input_ids.to(bilstm.device),
            numeric,
            categorical,
            attention_mask=attention_mask.to(bilstm.device)
        )
        probs = torch.softmax(logits, dim=1).cpu().numpy()   # (1, 6)

    return probs


def get_model_probabilities(text, models):
    """
    Build the 18-dim Stage-2 feature vector for the SVM meta-learner.

    x = [p_AraBERT | p_CamelBERT-MSA | p_BiLSTM]  ∈ R^18
    (6 softmax probs per base model × 3 models)
    """
    feature_parts = []

    # ---- 1. AraBERT v2 (6 probs) ----
    if models.get('arabert_model') and models.get('arabert_tokenizer'):
        try:
            probs = get_softmax_probs_from_classifier(
                text, models['arabert_tokenizer'], models['arabert_model']
            )
        except Exception as e:
            st.warning(f"AraBERT inference error: {e}")
            probs = np.zeros((1, 6))
    else:
        probs = np.zeros((1, 6))
    feature_parts.append(probs)

    # ---- 2. CamelBERT-MSA fine-tuned classifier (6 probs) ----
    if models.get('camelbert_model') and models.get('camelbert_tokenizer'):
        try:
            probs = get_softmax_probs_from_classifier(
                text, models['camelbert_tokenizer'], models['camelbert_model']
            )
        except Exception as e:
            st.warning(f"CamelBERT inference error: {e}")
            probs = np.zeros((1, 6))
    else:
        probs = np.zeros((1, 6))
    feature_parts.append(probs)

    # ---- 3. BiLSTM + internal CamelBERT-MSA backbone (6 probs) ----
    if models.get('bilstm_model') and models.get('bilstm_tokenizer'):
        try:
            probs = get_bilstm_probs(text, models)
        except Exception as e:
            st.warning(f"BiLSTM inference error: {e}")
            probs = np.zeros((1, 6))
    else:
        probs = np.zeros((1, 6))
    feature_parts.append(probs)

    # Concatenate → (1, 18)
    feature_vector = np.concatenate(feature_parts, axis=1)
    return feature_vector


# -----------------------------------------
# UI Layout
# -----------------------------------------
# Load images
logo_b64 = get_image_base64("logo2.png")
bg_b64   = get_image_base64("jamal.jpg")

st.markdown(f"""
    <style>
    /* ── Background ── */
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}

    /* Optional dark overlay so text stays readable */
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.45);
        z-index: 0;
    }}

    /* Ensure content sits above overlay */
    .block-container {{
        position: relative;
        z-index: 1;
    }}

    /* ── Text area RTL ── */
    textarea {{
        direction: rtl;
        text-align: right;
        font-size: 16px;
    }}

    /* ── Simplified result box ── */
    .simplified-box {{
        background-color: #1e3a2e;
        padding: 20px;
        border-radius: 10px;
        direction: rtl;
        text-align: right;
        color: #ffffff;
        border-right: 4px solid #4ade80;
        margin-top: 20px;
    }}

    /* ── Logo container ── */
    .logo-container {{
        display: flex;
        justify-content: center;
        margin-bottom: 10px;
    }}

    .logo-container img {{
        height: 90px;
        object-fit: contain;
    }}
    </style>

    <!-- Logo -->
    <div class="logo-container">
        <img src="data:image/png;base64,{logo_b64}" alt="logo"/>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <h3 style='text-align: center; direction: rtl; color: #e2e8f0;'>مصنف وتبسيط النصوص العربية</h3>
""", unsafe_allow_html=True)

st.markdown("---")

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
if st.button("📊 بَيِّنْ", use_container_width=True, type="primary"):
    if not text.strip():
        st.warning("⚠️ الرجاء إدخال نص.")
    elif not meta_learner:
        st.error("⚠️ لم يتم تحميل النموذج.")
    else:
        with st.spinner("جاري التحليل..."):
            cleaned = normalize_ar(text)

            features   = get_model_probabilities(cleaned, base_models)
            prediction = meta_learner.predict(features)[0]

            try:
                probs      = meta_learner.predict_proba(features)[0]
                confidence = probs[prediction - 1] if prediction <= len(probs) else np.max(probs)
            except Exception:
                confidence = 1.0

            st.session_state.classification_done = True
            st.session_state.readability_level   = prediction
            st.session_state.confidence          = confidence
            st.session_state.original_text       = text

# -----------------------------------------
# Display Results
# -----------------------------------------
if st.session_state.classification_done:
    st.markdown("---")
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

    # -----------------------------------------
    # بَسِّطْ Button (for levels 4–6)
    # -----------------------------------------
    if level >= 4:
        st.markdown("---")
        st.info("💡 هذا النص صعب القراءة. يمكنك تبسيطه بالضغط على الزر أدناه.")

        if st.button("✨ بَسِّطْ", use_container_width=True, type="secondary"):
            if simplifier_model and simplifier_tokenizer:
                with st.spinner("جاري التبسيط..."):
                    cleaned = normalize_ar(st.session_state.original_text)

                    inputs = simplifier_tokenizer(
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

                    st.markdown("---")
                    st.subheader("✨ النص المبسط")
                    st.markdown(
                        f'<div class="simplified-box">{simplified_text}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.warning("⚠️ نموذج التبسيط غير متوفر حالياً.")

st.markdown("---")
st.caption("© 2025 — مشروع بَيِّنْ")
