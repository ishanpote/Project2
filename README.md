# 📱 Telecom Customer Churn Prediction & Feature Explainability Engine

**Predictive Operational Modeling & Machine Learning Ingestion Pipeline**

This repository houses an end-to-end, production-grade machine learning classification pipeline designed to proactively predict subscriber attrition (*churn*) using the official multi-dimensional IBM Telco Dataset (7,043 historical customer profiles).

By implementing an imbalance-aware modeling architecture, the framework optimizes decision boundaries to aggressively prioritize the identification of flight-risk subscribers. It then automatically parses account telemetry variables into clear, interpretative business strategies and strategic customer portfolios.

---

## 🚀 Key Architectural Features

- **Automated Ingestion Layer:** Programmable hook that securely fetches, standardizes, and mirrors the official IBM Telco customer ledger directly to local workspaces.
- **Robust Feature Engineering:** Cleans implicit data gaps within continuous tracking attributes (`TotalCharges`) via median target imputation and maps textual variables using sparse one-hot binary vectorization.
- **Cost-Sensitive Learning Optimization:** Deploys an ensemble Random Forest algorithm tuned with balanced subsampling weights (`class_weight='balanced_subsample'`) to directly counter real-world subscriber distribution imbalances.
- **Explainable AI Engine:** Isolates global Gini impurity metrics to score, rank, and visibly map the top operational drivers behind account cancellation.
- **Dynamic Retention Segmentation:** Replaces flat binary outputs with a custom probability routing engine that partitions subscribers into actionable business segments (*At Risk*, *Loyal*, *Dormant*, *Standard Active*).

---

## 📊 Pipeline Diagnostic Outputs & Performance Benchmarks

### 1) Cost-Sensitive Model Performance

By penalizing the misclassification of the minority churn class, the framework maximizes true positive flags, shifting active subscriber recall up to **73%** to protect core enterprise margins.

```text
✓ Stratified Testing Accuracy Score: 76.79%

✓ Optimized Classification Report Summary:
              precision    recall  f1-score   support

           0       0.89      0.78      0.83      1035
           1       0.55      0.73      0.63       374

    accuracy                           0.77      1409
   macro avg       0.72      0.76      0.73      1409
weighted avg       0.80      0.77      0.78      1409
```

Additional diagnostics exported by the notebook:

- **Confusion matrix heatmap**
- **Global operational drivers (XAI Gini feature importance chart)**

### 2) Algorithmic Customer Segmentation Manifest

The evaluation engine maps the validation ledger directly into operational portfolios cached inside `outputs/customer_segments_manifest.csv` for targeted marketing CRM campaigns:

- 📦 **Loyal Portfolio:** 549 accounts — high longevity, low-probability secure enterprise assets.
- 📦 **At-Risk Portfolio:** 387 accounts — high model risk indicators (≥ 60% probability), high-priority targets.
- 📦 **Standard Active Portfolio:** 377 accounts — normal baseline subscribers running within stable cycles.
- 📦 **Dormant Portfolio:** 96 accounts — low activity signatures coupled with early life-cycle tenure states.

---

## 📁 Repository Structure

```text
telecom-churn-intelligence/
│
├── data/
│   └── Telco-Customer-Churn.csv         # Cached copy of official IBM dataset
│
├── src/
│   └── churn_notebook.ipynb             # Interactive primary notebook pipeline
│
├── outputs/
│   ├── churn_confusion_matrix.png       # Exported Seaborn performance heatmap
│   ├── churn_feature_importance.png     # Exported Gini contribution chart
│   └── customer_segments_manifest.csv   # Mapped subscriber account manifest
│
└── README.md                            # Repository master documentation
```

---

## 🛠️ Environment Initialization & Requirements

- **Core Runtime Environment:** Python 3.8+ (development verified on Python 3.11)
- **Required Core Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`

### Standard Local Dependency Virtual Setup

```bash
# Initialize clean virtual environment
python -m venv .venv

# Activate environment layer
source .venv/bin/activate  # Windows Terminal: .venv\Scripts\activate

# Upgrade dependency managers and install library array
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

## 🏃‍♂️ Execution Blueprint

> ⚠️ **CRITICAL RUNTIME REQUIREMENT:** To ensure notebook relative paths (`../data`, `../outputs`) resolve correctly without path errors, always initialize your Jupyter workspace from the repository root directory.

### 1) Interactive Development Execution

```bash
# Open notebook canvas via root terminal boundary
jupyter notebook src/churn_notebook.ipynb
```

Then select **Run All** from the notebook toolbar. The automated pipeline cells will sequentially execute ingestion, transform dimensions, compile diagnostics, and write assets back to `outputs/`.

### 2) Headless Production Execution (Automated Script Run)

To trigger the machine learning pipeline via terminal and export an updated executed notebook artifact:

```bash
jupyter nbconvert --to notebook --execute src/churn_notebook.ipynb --output executed_notebook.ipynb
```

---

## 💡 Future Pipeline Hardening Ideas

- **Hyperparameter Search Space Optimization:** Expand model tuning grids using cross-validated `RandomizedSearchCV`.
- **Alternative Boosting Architectures:** Benchmark Random Forest baselines against gradient-boosted models (XGBoost, LightGBM) and compare ROC/AUC.
- **Local Interpretability Upgrades:** Integrate SHAP values or LIME to extract precise, feature-level rationale for individual account profiles.

---

## 📄 License & Ownership

- **Author Profile:** Ishan Pote (GitHub Portfolio Interface)
- **Open-Source Tracking:** Pull requests and version-hardening submissions are welcome via open issues.
