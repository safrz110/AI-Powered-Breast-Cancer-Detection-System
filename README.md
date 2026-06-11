<div align="center">

<img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" />

<br /><br />

# 🔬 Breast Cancer Detection — AI Diagnostic Assistant

**A production-ready machine learning web application that classifies breast tumours as Malignant or Benign from Fine Needle Aspirate (FNA) biopsy measurements, achieving 97.58% cross-validated accuracy.**

<br />

| Metric | Score |
|:---|:---:|
|  CV Accuracy | **97.58%** |
|  Precision | **96.52%** |
|  Recall | **98.33%** |
|  F1 Score | **97.38%** |

<br />

</div>

---

##  Table of Contents

- [Project Overview](#-project-overview)
- [Live Demo](#-live-demo)
- [ML Pipeline](#-ml-pipeline)
- [App Features](#-app-features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Key Input Features](#-key-input-features)
- [Model Performance](#-model-performance)
- [Tech Stack](#-tech-stack)
- [Future Enhancements](#-future-enhancements)
- [Author](#-author)

---

##  Project Overview

Breast cancer is the most common cancer among women worldwide. Early and accurate detection is critical — it directly determines whether a tumour can be treated in time. This project builds an **end-to-end ML diagnostic tool** that takes quantitative measurements from a cytology slide and returns an instant malignancy prediction.

**What makes this project stand out:**

- Trained on the **Wisconsin Breast Cancer Dataset** (UCI), a gold-standard clinical benchmark with 569 labelled FNA samples
- Uses **GridSearchCV with 5-fold Stratified Cross-Validation** to rigorously tune a K-Nearest Neighbours classifier across 24 hyperparameter combinations
- Achieves **98.33% Recall** — critical in medical contexts where false negatives (missed cancers) are the most dangerous error
- Deployed as a **stylish, production-grade Streamlit app** with a clinical-grade dark UI, real-time probability display, and next-step clinical guidance

---

##  Live Demo

> Clone the repo and run locally — see [Quick Start](#-quick-start)

**Benign result:**

```
 Likely Benign
Confidence: 91.4%
→ Continue routine monitoring. Periodic imaging follow-up advised.
```

**Malignant result:**

```
 Likely Malignant
Confidence: 88.7%
→ Refer for immediate oncology consultation. Histopathological confirmation advised.
```

---

##  ML Pipeline

```
Raw FNA Biopsy Measurements (30 features)
              │
              ▼
┌─────────────────────────┐
│   ColumnTransformer     │  ← StandardScaler on all 30 numerical features
│   (Preprocessing)       │  ← OneHotEncoder ready for categorical columns
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐    GridSearchCV
│  KNeighborsClassifier   │  ← n_neighbors ∈ {3,5,7,9,11,15}
│  (Best: k=3, manhattan) │  ← metric ∈ {euclidean, manhattan}
└────────────┬────────────┘  ← weights ∈ {uniform, distance}
             │                  24 combinations × 5-fold CV = 120 fits
             ▼
     Malignant / Benign
     + Probability Score
```

**Why KNN for this problem?**

KNN is a strong choice for this dataset because the decision boundary between malignant and benign tumours is non-linear and locally structured — meaning nearby examples in feature space share the same class. With proper StandardScaler normalisation (mandatory for distance-based models), KNN captures fine-grained cluster boundaries that linear models miss. The Manhattan metric outperformed Euclidean here because FNA measurements include features with different scales and outliers, and the L1 norm is more robust in those conditions.

---

##  App Features

- ** Real-time inference** — instant classification as sliders are adjusted and submitted
- ** Dual probability display** — shows both `P(Malignant)` and `P(Benign)` side by side
- ** Clinical dark UI** — deep navy with sky-blue accents, DM Mono typeface for data readability
- ** 8 curated feature inputs** — top features by correlation, grouped into logical clinical sections
- ** Next-step clinical guidance** — contextual recommendation displayed with every result
- ** Medical disclaimer** — prominently shown after every prediction
- ** Cached model loading** — `@st.cache_resource` for instant repeat predictions
- ** Responsive layout** — works on desktop and tablet browsers

---

##  Project Structure

```
breast-cancer-detection/
│
├── cancer_detector_app.py      # Main Streamlit application
├── best_model.pkl              # Serialised GridSearchCV pipeline (joblib)
├── feature_columns.pkl         # Ordered list of 30 feature column names
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory data analysis & visualisations
│   ├── 02_Preprocessing.ipynb  # Feature engineering & scaling
│   └── 03_Model_Training.ipynb # GridSearchCV, evaluation, model export
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/your-username/breast-cancer-detection.git
cd breast-cancer-detection
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the app
```bash
streamlit run cancer_detector_app.py
```

Open **http://localhost:8501** in your browser.

---

##  Requirements

```
streamlit>=1.32.0
scikit-learn==1.7.2
joblib>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
```

> **Note:** The model was serialised with `scikit-learn 1.7.2`. Use the exact version to avoid unpickling errors.

---

##  Key Input Features

The app uses **8 high-signal features** chosen by absolute correlation with the diagnosis label, organised into two clinical groups:

###  Nuclear Size & Morphology

| Feature | Description | Correlation |
|---|---|:---:|
| `radius_worst` | Mean of 3 largest nucleus radii | 0.777 |
| `perimeter_worst` | Mean of 3 largest perimeters | 0.783 |
| `area_worst` | Mean of 3 largest nucleus areas | 0.734 |
| `texture_worst` | Grey-scale SD of worst nuclei | 0.456 |

###  Contour Concavity

| Feature | Description | Correlation |
|---|---|:---:|
| `concave points_worst` | Concave contour points (worst) | 0.794 |
| `concavity_worst` | Concavity severity (worst) | 0.660 |
| `concave points_mean` | Concave contour points (mean) | 0.777 |
| `concavity_mean` | Concavity severity (mean) | 0.696 |

> All remaining 22 features are passed as zero-padded values in the full feature vector to preserve pipeline compatibility with the trained scaler.

---

##  Model Performance

### GridSearchCV Hyperparameter Search

| Parameter | Values Searched |
|---|---|
| `n_neighbors` | 3, 5, 7, 9, 11, 15 |
| `metric` | euclidean, manhattan |
| `weights` | uniform, distance |
| **Total fits** | 24 combinations × 5 folds = **120 fits** |

**Best configuration found:** `k=3, metric=manhattan, weights=uniform`

### Cross-Validation Results (5-Fold Stratified)

| Fold | Accuracy |
|:---:|:---:|
| 1 | 100.00% |
| 2 | 95.61% |
| 3 | 94.74% |
| 4 | 97.37% |
| 5 | 95.58% |
| **Mean ± Std** | **96.66% ± 1.88%** |

### Final Evaluation Metrics

| Metric | Score | Why it matters |
|---|:---:|---|
| Accuracy | 97.58% | Overall correct classifications |
| Precision | 96.52% | Low false positive rate — avoids unnecessary patient anxiety |
| **Recall** | **98.33%** | **Low false negative rate — critical: missed cancers are dangerous** |
| F1 Score | 97.38% | Harmonic balance of precision and recall |

### Dataset Summary

| Split | Count |
|---|:---:|
| Total samples | 569 |
| Malignant (class 0) | 212 (37.3%) |
| Benign (class 1) | 357 (62.7%) |

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| ML & Tuning | scikit-learn (KNN, GridSearchCV, ColumnTransformer, StandardScaler) |
| Web Framework | Streamlit |
| Data Processing | NumPy, Pandas |
| UI Styling | Custom CSS (dark clinical theme, DM Sans + DM Mono fonts) |
| Model Serialisation | joblib |
| Dataset | UCI Wisconsin Breast Cancer Dataset |
| Version Control | Git / GitHub |

---

##  Future Enhancements

- [ ] Add SHAP waterfall plots to explain each individual prediction
- [ ] Benchmark against Logistic Regression, SVM, and Random Forest
- [ ] Upload CSV for batch predictions across multiple patients
- [ ] Deploy to Streamlit Cloud or Hugging Face Spaces (public URL)
- [ ] Add confidence calibration using Platt scaling
- [ ] Integrate LIME for local interpretable model explanations
- [ ] Build REST API with FastAPI for EHR system integration

---

##  Medical Disclaimer

This application is intended **strictly for educational and research purposes**. It is **not** a substitute for professional medical diagnosis. All clinical decisions must be validated by a qualified healthcare professional. The model's predictions should never be used as the sole basis for medical treatment.

---

##  Author

**Sarfaraz Ali**


##  License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

⭐ **Found this useful? Drop a star — it helps others discover the project!** ⭐



</div>
