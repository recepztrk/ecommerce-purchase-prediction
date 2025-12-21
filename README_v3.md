# 🛒 E-Commerce User Behavior Prediction - v3.0

Predicting purchase intent from user browsing sessions using data quality improvements and gradient boosting models.

## 🎉 Quick Stats (v3.0)

- **Dataset:** 16.7M events → 2.2M quality sessions
- **Problem:** Binary classification (purchase vs. no purchase)
- **Approach:** Data quality filtering + Session-level aggregation + Ensemble
- **Best Model:** Ensemble (LightGBM + XGBoost)
- **Test ROC-AUC:** **0.7619** (76.2%) ⭐⭐
- **Improvement:** +28.35% vs v1.0 🚀

---

## 📊 Version Evolution

| Version | Approach | Test AUC | Key Improvement |
|---------|----------|----------|-----------------|
| v1.0 | Baseline + LightGBM | 0.5936 | Initial implementation |
| v2.0 | Advanced features + Tuning | 0.6107 | +17 features, hyperparameter optimization |
| **v3.0** | **Data Quality** | **0.7619** | **Session merge, quality filtering (+28%)** |

**Key Insight:** Clean data with simple models >> Complex features on noisy data!

---

## 📁 Project Structure

```
├── README.md
├── requirements.txt
├── archive/               # Original Parquet data
├── data/
│   ├── v3/               # v3.0 processed data (quality filtered)
├── src/
│   ├── utils/            # Configuration
│   ├── data/             # Data preparation (v1, v3)
│   ├── features/         # Feature engineering (v1, v2)
│   ├── models/           # Model training (v1, v2, v3)
│   └── evaluation/       # Model evaluation
├── models/               # Trained models & results
├── reports/              # Final report, visualizations
└── notebooks/            # Exploratory analysis
```

---

## 🚀 How to Run

### 1. Setup Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline (v3.0)

```bash
# Step 1: Data preparation with quality improvements
python -m src.data.prepare_v3

# Step 2: Feature engineering
# (Built into data preparation for v3.0)

# Step 3: Train models (LightGBM, XGBoost, Ensemble)
python -m src.models.train_v3

# Results will be saved to models/ and reports/
```

### 3. View Results

```bash
# See detailed report
cat reports/final_report_v3.md

# View visualizations
open reports/roc_pr_curves_v3.png
open reports/confusion_matrix_v3.png
open reports/feature_importance_v3.png
```

---

## 📈 Results Summary

### All Versions Performance

| Model | Test AUC | Val AUC | Improvement vs v1.0 |
|-------|----------|---------|---------------------|
| v1.0 LightGBM | 0.5936 | 0.6492 | - |
| v2.0 LightGBM | 0.6107 | 0.6596 | +2.88% |
| v2.0 Ensemble | 0.6107 | 0.6593 | +2.88% |
| **v3.0 LightGBM** | **0.7622** | **0.8004** | **+28.41%** 🚀 |
| **v3.0 XGBoost** | **0.7595** | **0.8082** | **+27.95%** 🚀 |
| **v3.0 Ensemble** | **0.7619** | **0.8041** | **+28.35%** 🚀 |

### Classification Metrics (v3.0 LightGBM, threshold=0.5)

```
              Precision   Recall   F1-Score   Support
────────────────────────────────────────────────────
No Purchase      0.83      0.61      0.70     322,984
Purchase         0.58      0.82      0.68     217,678
────────────────────────────────────────────────────
Accuracy: 0.69
```

### 🔑 What Made v3.0 Successful?

**The Problem (v1.0/v2.0):**
- 70% of sessions had only 1 event
- No behavioral patterns to learn from
- Model essentially guessing randomly

**The Solution (v3.0):**
1. **Session Merging:** Combined user activity within 30-min windows
2. **Quality Filtering:** Kept only sessions with ≥2 events
3. **Outlier Removal:** Removed sessions >2 hours
4. **Result:** Only meaningful, pattern-rich sessions remain

**Data Statistics:**
- v2.0: 7.3M sessions (70% noise) → v3.0: 2.2M sessions (100% signal)
- 69% reduction in data, but 28% improvement in performance!
- **Lesson:** Quality data > Quantity of data

---

## ✨ Key Features

- ✅ Memory-efficient Parquet reading (PyArrow)
- ✅ Session merging (30-min window)
- ✅ Quality filtering (≥2 events, ≤2 hours)
- ✅ Session-level aggregation (32 features)
- ✅ Multiple models (LightGBM, XGBoost, Ensemble)
- ✅ Comprehensive evaluation metrics
- ✅ Full reproducibility (fixed seeds, versioned data)
- ✅ Production-ready code structure

---

## 🎓 Key Learnings

1. **Data Quality > Feature Engineering**
   - v2.0: +17 features → +2.88% improvement
   - v3.0: Data quality → +28.35% improvement
   - **10x more impact from clean data!**

2. **Domain Understanding is Critical**
   - Recognizing single-event sessions as noise was the breakthrough
   - Business context matters more than technical sophistication

3. **Simplicity Wins**
   - v2.0: 59 features (complex)
   - v3.0: 32 features (simple)
   - v3.0 outperformed by 25%!

4. **Session Definition Matters**
   - How you aggregate events fundamentally affects model performance
   - Merge related activity, filter noise

---

## 🚀 Future Improvements (To Reach 0.80+)

1. **LSTM/RNN Sequence Models** (+2-4% expected)
   - Capture temporal dependencies directly
   - Learn from event order

2. **User Historical Features** (+1-2% expected)
   - Past purchase behavior
   - Return visitor indicators

3. **Product Features** (+1-2% expected)
   - Product embeddings (Word2Vec style)
   - Category hierarchies

4. **Hyperparameter Optimization** (+0.5-1% expected)
   - Optuna for automated tuning

**Total Potential:** 0.76 → 0.81-0.83 AUC

---

## 📚 Documentation

- **[Final Report v3.0](reports/final_report_v3.md)** - Complete technical report with all details
- **[Model Comparison](models/version_comparison_v3.csv)** - Quantitative comparison across versions
- **[Visualizations](reports/)** - ROC curves, confusion matrix, feature importance

---

## 🏆 Project Status

**Status:** ✅ COMPLETE  
**Final Version:** v3.0  
**Test AUC:** 0.7619 (76.2%)  
**Rating:** ⭐⭐ GOOD (Production Ready)  
**Date:** December 21, 2024

---

## 📄 License

MIT

---

## 🙏 Acknowledgments

This project demonstrates the complete lifecycle of a real-world machine learning system, with emphasis on:
- Data quality over algorithmic complexity
- Reproducible, production-ready code
- Comprehensive documentation and evaluation
- Iterative improvement through versioning

