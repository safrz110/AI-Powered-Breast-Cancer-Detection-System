import streamlit as st
import numpy as np
import pandas as pd
import joblib, pickle, warnings, time
warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OncoSense – Breast Cancer Diagnostic & Prediction System",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #050A14 !important;
    color: #E2E8F0;
}

.block-container {
    padding: 2.5rem 1.5rem 4rem !important;
    max-width: 780px !important;
}

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 2.8rem 2rem 2rem;
    margin-bottom: 2rem;
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(56,189,248,0.12) 0%, transparent 70%);
    border-bottom: 1px solid rgba(56,189,248,0.1);
}
.app-header .eyebrow {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #38BDF8;
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.2);
    padding: 0.3rem 0.9rem;
    border-radius: 100px;
    margin-bottom: 1rem;
}
.app-header h1 {
    font-size: 2.1rem;
    font-weight: 700;
    line-height: 1.15;
    color: #F1F5F9;
    margin: 0 0 0.6rem;
    letter-spacing: -0.5px;
}
.app-header h1 span { color: #38BDF8; }
.app-header p {
    color: #64748B;
    font-size: 0.92rem;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}
.stat-strip {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    margin-top: 1.8rem;
    flex-wrap: wrap;
}
.stat-item { text-align: center; }
.stat-value {
    display: block;
    font-size: 1.5rem;
    font-weight: 700;
    color: #38BDF8;
    font-family: 'DM Mono', monospace;
}
.stat-label {
    display: block;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #475569;
    margin-top: 0.15rem;
}

/* ── Section cards ── */
.section-wrap {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.6rem 1.8rem 1.2rem;
    margin-bottom: 1.2rem;
}
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #38BDF8;
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(56,189,248,0.15);
}

/* ── Sliders ── */
.stSlider > label {
    font-size: 0.83rem !important;
    color: #94A3B8 !important;
    font-weight: 500 !important;
}
.stSlider [data-baseweb="slider"] {
    padding-top: 0.3rem !important;
}
.stSlider [data-testid="stThumbValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    background: #1E293B !important;
    color: #38BDF8 !important;
    border: 1px solid rgba(56,189,248,0.3) !important;
    border-radius: 6px !important;
}

/* ── Button ── */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0EA5E9 0%, #2563EB 100%);
    color: white !important;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.85rem 2rem;
    border: none;
    border-radius: 12px;
    cursor: pointer;
    letter-spacing: 0.2px;
    box-shadow: 0 0 30px rgba(14,165,233,0.25), 0 4px 15px rgba(0,0,0,0.3);
    transition: all 0.2s ease;
    margin-top: 0.8rem;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 45px rgba(14,165,233,0.4), 0 8px 25px rgba(0,0,0,0.35);
}

/* ── Results ── */
.result-benign {
    background: linear-gradient(135deg, #064E3B 0%, #065F46 100%);
    border: 1px solid #10B981;
    border-radius: 18px;
    padding: 2.2rem 2rem;
    text-align: center;
    box-shadow: 0 0 50px rgba(16,185,129,0.15), 0 8px 30px rgba(0,0,0,0.3);
    margin: 1.5rem 0;
}
.result-malignant {
    background: linear-gradient(135deg, #450A0A 0%, #7F1D1D 100%);
    border: 1px solid #EF4444;
    border-radius: 18px;
    padding: 2.2rem 2rem;
    text-align: center;
    box-shadow: 0 0 50px rgba(239,68,68,0.15), 0 8px 30px rgba(0,0,0,0.3);
    margin: 1.5rem 0;
}
.result-icon { font-size: 2.8rem; margin-bottom: 0.5rem; }
.result-verdict {
    font-size: 1.75rem;
    font-weight: 700;
    color: white;
    letter-spacing: -0.3px;
    margin: 0;
}
.result-confidence {
    font-family: 'DM Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    color: white;
    margin: 0.5rem 0 0.3rem;
    line-height: 1;
}
.result-sub {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.6);
    margin-top: 0.2rem;
}
.result-action {
    margin-top: 1.2rem;
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: rgba(255,255,255,0.8);
    line-height: 1.5;
}

/* ── Metrics row ── */
.metrics-row {
    display: flex;
    gap: 0.8rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.metric-chip {
    flex: 1;
    min-width: 130px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
.chip-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #475569;
    display: block;
    margin-bottom: 0.25rem;
}
.chip-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.05rem;
    font-weight: 600;
    color: #E2E8F0;
}

/* ── Disclaimer ── */
.disclaimer {
    margin-top: 1.5rem;
    padding: 0.9rem 1.1rem;
    background: rgba(245,158,11,0.06);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 10px;
    font-size: 0.78rem;
    color: #B45309;
    line-height: 1.5;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #1E293B;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Hide Streamlit default UI */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model  = joblib.load("best_model.pkl")
    with open("feature_columns.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

model, all_features = load_artifacts()


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="eyebrow">🔬 ONCOSENSE AI PLATFORM</div>
<h2>OncoSense<br><span>Breast Cancer Diagnostic & Prediction System</span></h2>
<p>Enter cell nucleus measurements from a fine needle aspirate biopsy.
   The AI model provides an instant breast cancer diagnostic prediction and malignancy risk assessment.</p>
    <div class="stat-strip">
        <div class="stat-item">
            <span class="stat-value">97.6%</span>
            <span class="stat-label">Model Accuracy</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">KNN</span>
            <span class="stat-label">Algorithm</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">569</span>
            <span class="stat-label">Training Samples</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">8</span>
            <span class="stat-label">Key Features</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Feature config  (top 8 by correlation, mean default, real ranges) ───────
FEATURES = {
    # ── Size & Shape (Worst)
    "radius_worst": {
        "label": "Radius — Worst  (mean of largest radii)",
        "min": 7.0, "max": 37.0, "default": 16.3, "step": 0.1,
        "help": "Mean of the three largest cell nucleus radii in the sample.",
        "group": "size",
    },
    "perimeter_worst": {
        "label": "Perimeter — Worst",
        "min": 50.0, "max": 255.0, "default": 107.0, "step": 0.5,
        "help": "Mean of the three largest nucleus perimeters.",
        "group": "size",
    },
    "area_worst": {
        "label": "Area — Worst  (μm²)",
        "min": 180.0, "max": 4300.0, "default": 880.0, "step": 5.0,
        "help": "Mean of the three largest nucleus areas.",
        "group": "size",
    },
    "texture_worst": {
        "label": "Texture — Worst  (grey-scale SD)",
        "min": 12.0, "max": 50.0, "default": 25.7, "step": 0.1,
        "help": "Standard deviation of grey-scale values, worst nuclei.",
        "group": "size",
    },
    # ── Concavity & Shape (Worst + Mean)
    "concave points_worst": {
        "label": "Concave Points — Worst",
        "min": 0.0, "max": 0.30, "default": 0.114, "step": 0.001,
        "help": "Number of concave portions of the contour (worst nuclei).",
        "group": "shape",
    },
    "concavity_worst": {
        "label": "Concavity — Worst  (severity)",
        "min": 0.0, "max": 1.30, "default": 0.27, "step": 0.005,
        "help": "Severity of concave portions of the contour (worst nuclei).",
        "group": "shape",
    },
    "concave points_mean": {
        "label": "Concave Points — Mean",
        "min": 0.0, "max": 0.21, "default": 0.048, "step": 0.001,
        "help": "Average number of concave contour portions across all nuclei.",
        "group": "shape",
    },
    "concavity_mean": {
        "label": "Concavity — Mean",
        "min": 0.0, "max": 0.45, "default": 0.089, "step": 0.002,
        "help": "Average severity of concave contour portions.",
        "group": "shape",
    },
}

# ─── Input UI ────────────────────────────────────────────────────────────────
user_inputs = {}

# Group A: Size & Morphology
st.markdown('<div class="section-wrap"><div class="section-label">📐 Nuclear Size & Morphology</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
size_feats = [k for k, v in FEATURES.items() if v["group"] == "size"]
for idx, key in enumerate(size_feats):
    meta = FEATURES[key]
    col = c1 if idx % 2 == 0 else c2
    with col:
        user_inputs[key] = st.slider(
            meta["label"], meta["min"], meta["max"],
            meta["default"], meta["step"], help=meta["help"],
            key=key
        )
st.markdown('</div>', unsafe_allow_html=True)

# Group B: Concavity & Contour
st.markdown('<div class="section-wrap"><div class="section-label">🔵 Contour Concavity Measurements</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
shape_feats = [k for k, v in FEATURES.items() if v["group"] == "shape"]
for idx, key in enumerate(shape_feats):
    meta = FEATURES[key]
    col = c3 if idx % 2 == 0 else c4
    with col:
        user_inputs[key] = st.slider(
            meta["label"], meta["min"], meta["max"],
            meta["default"], meta["step"], help=meta["help"],
            key=key
        )
st.markdown('</div>', unsafe_allow_html=True)

# ─── Predict Button ───────────────────────────────────────────────────────────
run = st.button("🔬 Predict Breast Cancer Risk", use_container_width=True)

if run:
    # Build full feature vector (zeros for non-selected features)
    row = {feat: 0.0 for feat in all_features}
    row.update(user_inputs)
    X_input = pd.DataFrame([row], columns=all_features)

    with st.spinner("OncoSense AI is analyzing the sample..."):
        time.sleep(0.6)   # brief pause for UX
        pred       = model.predict(X_input)[0]          # 0=Malignant, 1=Benign
        proba      = model.predict_proba(X_input)[0]    # [P(Mal), P(Ben)]

    p_malignant = proba[0] * 100
    p_benign    = proba[1] * 100
    label       = "Malignant" if pred == 0 else "Benign"
    confidence  = p_malignant if pred == 0 else p_benign

    # ── Result card ──
    if pred == 0:
        st.markdown(f"""
        <div class="result-malignant">
            <div class="result-icon">⚠️</div>
            <p class="result-verdict">Likely Malignant</p>
            <p class="result-confidence">{confidence:.1f}%</p>
            <p class="result-sub">Model confidence score</p>
            <div class="result-action">
                🏥 <strong>Recommended next step:</strong> Refer for immediate oncology consultation.
                Histopathological biopsy confirmation is advised before clinical decisions.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-benign">
            <div class="result-icon">✅</div>
            <p class="result-verdict">Likely Benign</p>
            <p class="result-confidence">{confidence:.1f}%</p>
            <p class="result-sub">Model confidence score</p>
            <div class="result-action">
                📋 <strong>Recommended next step:</strong> Continue routine monitoring.
                Periodic imaging follow-up is still advised per clinical guidelines.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Key Metric chips ──
    chips_html = f"""
    <div class="metrics-row">
        <div class="metric-chip">
            <span class="chip-label">Prediction</span>
            <span class="chip-value">{'🔴 ' if pred==0 else '🟢 '}{label}</span>
        </div>
        <div class="metric-chip">
            <span class="chip-label">Malignant P</span>
            <span class="chip-value">{p_malignant:.1f}%</span>
        </div>
        <div class="metric-chip">
            <span class="chip-label">Benign P</span>
            <span class="chip-value">{p_benign:.1f}%</span>
        </div>
        <div class="metric-chip">
            <span class="chip-label">Radius Worst</span>
            <span class="chip-value">{user_inputs['radius_worst']:.1f} mm</span>
        </div>
    </div>
    """
    st.markdown(chips_html, unsafe_allow_html=True)

    # ── Disclaimer ──
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>OncoSense Medical Disclaimer:</strong> This AI-powered prediction system is intended for research and educational purposes only. It is not a substitute for professional medical diagnosis. All clinical
        decisions must be made by a qualified healthcare professional.
    </div>
    """, unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    OncoSense – Breast Cancer Diagnostic Prediction System · KNN Classifier · GridSearchCV Optimised · Wisconsin Breast Cancer Dataset
</div>
""", unsafe_allow_html=True)