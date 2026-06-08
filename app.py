import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OncoSense — Breast Cancer Detection & Classification System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&display=swap');

/* ── Root & Background ── */
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 40%, #0a1628 100%);
    color: #e8f4fd;
}

/* ── Header Banner ── */
.hero-banner {
    background: linear-gradient(90deg, #00b4d8 0%, #0077b6 40%, #023e8a 100%);
    border-radius: 18px;
    padding: 38px 48px;
    margin-bottom: 32px;
    box-shadow: 0 8px 40px rgba(0,180,216,0.25);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(255,255,255,0.07);
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 140px; height: 140px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -1px;
    margin: 0 0 6px 0;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.82);
    font-family: 'Space Mono', monospace;
    margin: 0;
}

/* ── Cards ── */
.info-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,180,216,0.18);
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 20px;
}
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #00b4d8;
    margin-bottom: 14px;
}

/* ── Result Boxes ── */
.result-malignant {
    background: linear-gradient(135deg, #3d0000, #7b0000);
    border: 2px solid #ff4444;
    border-radius: 18px;
    padding: 36px 40px;
    text-align: center;
    box-shadow: 0 0 40px rgba(255,68,68,0.3);
    animation: pulse-red 2s ease-in-out infinite;
}
.result-benign {
    background: linear-gradient(135deg, #003d1a, #006b2e);
    border: 2px solid #00e676;
    border-radius: 18px;
    padding: 36px 40px;
    text-align: center;
    box-shadow: 0 0 40px rgba(0,230,118,0.3);
    animation: pulse-green 2s ease-in-out infinite;
}
@keyframes pulse-red {
    0%,100% { box-shadow: 0 0 30px rgba(255,68,68,0.25); }
    50%      { box-shadow: 0 0 60px rgba(255,68,68,0.50); }
}
@keyframes pulse-green {
    0%,100% { box-shadow: 0 0 30px rgba(0,230,118,0.20); }
    50%      { box-shadow: 0 0 55px rgba(0,230,118,0.45); }
}
.result-label {
    font-size: 1rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
    opacity: 0.75;
    margin-bottom: 10px;
}
.result-value {
    font-size: 2.6rem;
    font-weight: 800;
    margin: 0;
}
.result-conf {
    font-size: 1rem;
    font-family: 'Space Mono', monospace;
    opacity: 0.7;
    margin-top: 8px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06101e 0%, #0a1628 100%);
    border-right: 1px solid rgba(0,180,216,0.15);
}
[data-testid="stSidebar"] .stSlider > div { color: #e8f4fd; }

/* ── Slider accent ── */
[data-testid="stSlider"] [role="slider"] {
    background: #00b4d8 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(90deg, #00b4d8, #0077b6);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 1px;
    padding: 14px 0;
    width: 100%;
    transition: all 0.2s ease;
    cursor: pointer;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #0096c7, #023e8a);
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(0,180,216,0.4);
}

/* ── Metric tiles ── */
.metric-tile {
    background: rgba(0,180,216,0.08);
    border: 1px solid rgba(0,180,216,0.22);
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
}
.metric-num  { font-size: 2rem; font-weight: 800; color: #00b4d8; }
.metric-desc { font-size: 0.8rem; opacity: 0.65; margin-top: 4px; }

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(255,193,7,0.08);
    border-left: 3px solid #ffc107;
    border-radius: 0 8px 8px 0;
    padding: 12px 18px;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.65);
    margin-top: 24px;
}

/* ── divider ── */
hr { border-color: rgba(0,180,216,0.15) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("Breast-Cancer-Detection-Classification-Analysis_Model.pkl")

model = load_model()

FEATURES = [
    "radius_mean","texture_mean","perimeter_mean","area_mean",
    "smoothness_mean","compactness_mean","concavity_mean","concave points_mean",
    "symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se",
    "smoothness_se","compactness_se","concavity_se","concave points_se",
    "symmetry_se","fractal_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst",
    "smoothness_worst","compactness_worst","concavity_worst","concave points_worst",
    "symmetry_worst","fractal_dimension_worst",
]

# Typical min/max ranges for the Wisconsin dataset (for slider bounds)
RANGES = {
    "radius_mean":         (6.9,  28.1),   "texture_mean":        (9.7,  39.3),
    "perimeter_mean":      (43.8, 188.5),  "area_mean":           (143.5,2501.0),
    "smoothness_mean":     (0.053,0.163),  "compactness_mean":    (0.019,0.345),
    "concavity_mean":      (0.0,  0.427),  "concave points_mean": (0.0,  0.201),
    "symmetry_mean":       (0.106,0.304),  "fractal_dimension_mean":(0.05,0.097),
    "radius_se":           (0.112,2.873),  "texture_se":          (0.36, 4.885),
    "perimeter_se":        (0.76, 21.98),  "area_se":             (6.8,  542.2),
    "smoothness_se":       (0.002,0.031),  "compactness_se":      (0.002,0.135),
    "concavity_se":        (0.0,  0.396),  "concave points_se":   (0.0,  0.053),
    "symmetry_se":         (0.008,0.079),  "fractal_dimension_se":(0.001,0.03),
    "radius_worst":        (7.9,  36.0),   "texture_worst":       (12.0, 49.5),
    "perimeter_worst":     (50.4, 251.2),  "area_worst":          (185.2,4254.0),
    "smoothness_worst":    (0.071,0.222),  "compactness_worst":   (0.027,1.058),
    "concavity_worst":     (0.0,  1.252),  "concave points_worst":(0.0,  0.291),
    "symmetry_worst":      (0.156,0.664),  "fractal_dimension_worst":(0.055,0.208),
}

DEFAULTS = {f: round((RANGES[f][0]+RANGES[f][1])/2, 4) for f in FEATURES}

# ─── Hero Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <p class="hero-title">🔬 OncoSense</p>
  <p class="hero-subtitle">Intelligent Breast Cancer Detection &amp; Classification · KNN Pipeline · Wisconsin Diagnostic Dataset</p>
</div>
""", unsafe_allow_html=True)

# ─── Top Stats Row ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, num, desc in zip(
    [c1,c2,c3,c4],
    ["30","KNN","2","~95%"],
    ["Input Features","Algorithm","Output Classes","Typical Accuracy"]
):
    with col:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-num">{num}</div>
            <div class="metric-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Sidebar Inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Feature Input Panel")
    st.markdown("Adjust the 30 tumor-cell measurements using the sliders below.")
    st.markdown("---")

    vals = {}

    groups = [
        ("📊 Mean Values", FEATURES[:10]),
        ("📉 Standard Error Values", FEATURES[10:20]),
        ("📈 Worst Values", FEATURES[20:]),
    ]

    for group_label, group_feats in groups:
        with st.expander(group_label, expanded=(group_label == "📊 Mean Values")):
            for feat in group_feats:
                lo, hi = RANGES[feat]
                step = round((hi - lo) / 200, 6)
                vals[feat] = st.slider(
                    feat.replace("_", " ").title(),
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(DEFAULTS[feat]),
                    step=step,
                    format="%.4f",
                )

    st.markdown("---")
    predict_btn = st.button("🔬 Run Diagnosis",width="stretch")

    st.markdown("""
    <div class="disclaimer">
    ⚠️ <b>Educational use only.</b> This tool is not a substitute for professional medical advice, diagnosis, or treatment.
    </div>""", unsafe_allow_html=True)

# ─── Main Panel ──────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div class="section-label">Feature Overview</div>', unsafe_allow_html=True)

    # Show a neat table of current values
    import pandas as pd
    df_display = pd.DataFrame({
        "Feature": [f.replace("_", " ").title() for f in FEATURES],
        "Value": [f"{vals[f]:.4f}" for f in FEATURES],
        "Group": (["Mean"]*10 + ["Std Error"]*10 + ["Worst"]*10)
    })

    tab1, tab2, tab3 = st.tabs(["📊 Mean", "📉 Std Error", "📈 Worst"])
    for tab, grp in zip([tab1, tab2, tab3], ["Mean", "Std Error", "Worst"]):
        with tab:
            subset = df_display[df_display["Group"] == grp].drop(columns="Group")
            st.dataframe(subset,width="stretch",hide_index=True)

with right:
    st.markdown('<div class="section-label">Diagnosis Result</div>', unsafe_allow_html=True)

    if predict_btn:
        with st.spinner("Analyzing tumor features..."):
            time.sleep(0.8)   # brief dramatic pause

        input_df = pd.DataFrame([[vals[f] for f in FEATURES]],columns=FEATURES)
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        benign_pct    = round(proba[0] * 100, 1)
        malignant_pct = round(proba[1] * 100, 1)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-malignant">
                <div class="result-label">Classification Result</div>
                <div class="result-value">⚠️ Malignant</div>
                <div class="result-conf">Confidence: {malignant_pct}%</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-benign">
                <div class="result-label">Classification Result</div>
                <div class="result-value">✅ Benign</div>
                <div class="result-conf">Confidence: {benign_pct}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability bar chart
        st.markdown('<div class="section-label">Probability Breakdown</div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({
            "Class": ["Benign (0)", "Malignant (1)"],
            "Probability (%)": [benign_pct, malignant_pct]
        })
        st.bar_chart(prob_df.set_index("Class"),color=["#00b4d8"],width="stretch")

        # Key features highlight
        st.markdown('<div class="section-label">Top Influential Inputs</div>', unsafe_allow_html=True)
        top3 = sorted(
            [("concavity_worst", vals["concavity_worst"]),
             ("concave points_worst", vals["concave points_worst"]),
             ("radius_worst", vals["radius_worst"]),
             ("perimeter_worst", vals["perimeter_worst"]),
             ("area_worst", vals["area_worst"])],
            key=lambda x: abs(x[1] - DEFAULTS[x[0]]) / (RANGES[x[0]][1]-RANGES[x[0]][0]),
            reverse=True
        )[:3]
        for feat, val in top3:
            lo, hi = RANGES[feat]
            pct = int((val - lo) / (hi - lo) * 100)
            st.markdown(f"""
            <div class="info-card" style="padding:14px 18px; margin-bottom:10px;">
                <b>{feat.replace('_',' ').title()}</b>
                <div style="margin-top:8px; background:rgba(255,255,255,0.08); border-radius:6px; height:8px;">
                    <div style="width:{pct}%; background: linear-gradient(90deg,#00b4d8,#0077b6); height:8px; border-radius:6px;"></div>
                </div>
                <span style="font-family:'Space Mono',monospace; font-size:0.82rem; color:#00b4d8;">{val:.4f}</span>
                <span style="font-size:0.75rem; opacity:0.5; margin-left:8px;">{pct}th percentile of range</span>
            </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="info-card" style="text-align:center; padding: 60px 30px;">
            <div style="font-size:3.5rem; margin-bottom:16px;">🔬</div>
            <div style="font-size:1.2rem; font-weight:700; color:#00b4d8; margin-bottom:8px;">Ready to Diagnose</div>
            <div style="opacity:0.6; font-size:0.9rem; line-height:1.6;">
                Adjust the 30 cell-nuclei measurements<br>in the sidebar, then click<br><b>Run Diagnosis</b>.
            </div>
        </div>""", unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; opacity:0.4; font-family:'Space Mono',monospace; font-size:0.75rem; padding: 8px 0 20px;">
OncoSense · Intelligent Cancer Classification · Wisconsin Diagnostic Breast Cancer Dataset · For educational purposes only
</div>""", unsafe_allow_html=True)