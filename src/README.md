# Source Code Directory (src/)

Bu klasör, projenin tüm kaynak kodunu içerir. Python modülleri ve scriptler burada organize edilmiştir.

---

## 📂 Klasör Yapısı

```
src/
├── __init__.py              # Python package tanımı
├── models/                  # Model eğitimi ve optimizasyon scriptleri
├── features/                # Feature engineering ve veri dönüşümleri
├── data/                    # Veri hazırlama ve preprocessing
├── evaluation/              # Model değerlendirme ve metrikler
├── analysis/                # Veri analizi ve görselleştirme
└── utils/                   # Yardımcı fonksiyonlar ve konfigürasyon
```

**Toplam:** 6 alt klasör, 26 Python dosyası (~5,100 satır kod)

---

## 🤖 models/ - Model Eğitimi (10 dosya, ~2,700 satır)

Model eğitimi, hyperparameter optimization ve ensemble çalışmaları.

### **Baseline Models**

#### `train_kfold.py` (329 satır) ⭐
- **Amaç:** K-fold cross-validation ile model eğitimi
- **Kullanım:** v3.0 baseline eğitimi
- **Çalıştırma:** `python -m src.models.train_kfold`
- **Çıktı:** `models/lightgbm_v3.txt`, `models/xgboost_v3.json`
- **Özellikler:**
  - 5-fold CV
  - LightGBM + XGBoost
  - Early stopping
  - Model serialization

#### `train_v3.py` (265 satır)
- **Amaç:** v3.0 model eğitimi (single run)
- **Kullanım:** Hızlı eğitim için
- **Fonksiyonlar:** `load_data_v3()`, `train_lgb()`, `train_xgb()`

#### `train.py` (336 satır)
- **Amaç:** Genel training pipeline
- **Durum:** Eski, train_kfold preferred
- **Not:** Referans için saklandı

---

### **Phase Optimization Scripts**

#### `phase1_step5_validation.py` (108 satır)
- **Amaç:** Phase 1 - Final clean data validation
- **İş:** Clean dataset (24 features) değerlendirmesi
- **Çıktı:** `reports/phase1_*.csv`

#### `phase2_algorithm_testing.py` (244 satır)
- **Amaç:** Phase 2 - 5 farklı algoritma test
- **Algoritmalar:** LightGBM, XGBoost, Random Forest, ExtraTrees, HistGradientBoosting
- **Çıktı:** `reports/phase2_algorithm_comparison.csv`

####`phase3_optuna_tuning.py` (243 satır)
- **Amaç:** Phase 3 - Hyperparameter optimization (Optuna)
- **Modeller:** Top 3 (LightGBM, XGBoost, ExtraTrees)
- **Trials:** 25/model
- **Çıktı:** `models/best_*.pkl`, `reports/phase3_*.csv`
- **Özellikler:**
  - Bayesian optimization
  - Early stopping
  - Best model save

#### `phase4_ensemble.py` (311 satır)
- **Amaç:** Phase 4 - Weighted voting ensemble
- **Yöntem:** Grid search for optimal weights
- **Çıktı:** `models/ensemble_phase4_weights.json`

#### `phase4b_alternative_ensemble.py` (212 satır)
- **Amaç:** Phase 4b - Equal weights ve stacking
- **Yöntemler:** 
  - Equal weights (0.33, 0.33, 0.33)
  - Stacking with LogisticRegression
- **Çıktı:** `reports/phase4b_*.csv`

#### `phase4c_multiobjective.py` (424 satır - EN BÜYÜK!)
- **Amaç:** Phase 4c - Multi-objective ensemble optimization
- **Objective Functions:** 6 farklı (AUC, F1, AUC+F1, etc.)
- **Yöntem:** Scipy optimize
- **Çıktı:** `reports/phase4c_multiobjective_results.csv`

#### `v3_hyperparameter_tuning.py` (205 satır)
- **Amaç:** v3.0'ı Optuna ile tuning (50 trials)
- **Sonuç:** Başarısız (overtuning)
- **Çıktı:** `models/v3_lightgbm_optimized.txt`

---

## 🔧 features/ - Feature Engineering (5 dosya, ~1,200 satır)

Veri transformasyonu ve feature oluşturma.

### `engineered_features.py` (268 satır) ⭐
- **Amaç:** Final feature engineering pipeline
- **Fonksiyonlar:**
  - `create_session_features()` - Session aggregation
  - `create_temporal_features()` - Zaman feature'ları
  - `create_engagement_features()` - Event rate, product diversity
  - `create_price_features()` - Fiyat istatistikleri
- **Çıktı:** Session-level dataframe (24 features)

### `advanced.py` (274 satır)
- **Amaç:** Advanced feature engineering (v1/v2 için)
- **Features:**
  - Event sequences
  - Category interactions
  - Temporal patterns
- **Durum:** Eski, reference için

### `advanced_v3.py` (275 satır)
- **Amaç:** v3.0 için advanced features
- **Fonksiyon:** `create_event_sequence_features()`
- **Features:**
  - Purchase funnel patterns
  - Shopping behavior sequences
- **Kullanım:** Phase optimizasyonlarında

### `build.py` (174 satır)
- **Amaç:** Feature builder utility
- **Kullanım:** Data pipeline'da
- **Fonksiyonlar:** Generic feature transformation helpers

---

## 📥 data/ - Data Preprocessing (3 dosya, ~780 satır)

Ham veriden session-level veriye dönüşüm.

### `prepare_v3.py` (269 satır) ⭐
- **Amaç:** v3.0 data preparation
- **İş Akışı:**
  1. Event-level data okuma (`archive/train.parquet`)
  2. Session aggregation
  3. Feature engineering
  4. Train/val/test split
  5. Save parquet files
- **Çıktı:** `data/v3/train_sessions_v3.parquet`
- **Çalıştırma:** `python -m src.data.prepare_v3`

### `prepare.py` (242 satır)
- **Amaç:** Original data preparation (v1/v2)
- **Durum:** Eski pipeline
- **Not:** Referans için saklandı

---

## 📊 evaluation/ - Model Evaluation (3 dosya, ~885 satır)

Model performansını değerlendirme ve görselleştirme.

### `detailed_metrics_phase3.py` (233 satır) ⭐
- **Amaç:** Phase 3 models için detaylı metrikler
- **Metrikler:**
  - AUC (train/val/test)
  - F1, Precision, Recall
  - Confusion Matrix
  - Train-test gap
- **Çıktı:** `reports/phase3_detailed_metrics.csv`

### `evaluate.py` (326 satır)
- **Amaç:** Genel evaluation pipeline
- **Fonksiyonlar:**
  - `calculate_metrics()` - Tüm metrikler
  - `plot_roc_curve()` - ROC curve
  - `plot_confusion_matrix()` - Confusion matrix
  - `plot_feature_importance()` - Feature importance
- **Sıktı:** Report dosyaları + görseller

---

## 🔍 analysis/ - Data Analysis (1 dosya)

### `feature_analysis.py` (349 satır)
- **Amaç:** Feature importance ve correlation analizi
- **Analizler:**
  - Feature importance (LightGBM/XGBoost)
  - Correlation heatmap
  - Statistical summary
  - Missing value analysis
  - Outlier detection
- **Çıktı:** 
  - `reports/feature_analysis/correlation_heatmap.png`
  - `reports/feature_analysis/feature_importance.png`
- **Kullanım:** Phase 1'de veri analizi için

---

## ⚙️ utils/ - Utilities (2 dosya)

### `config.py`
- **Amaç:** Global configuration
- **İçerik:**
  - Data paths
  - Model hyperparameters
  - Random seeds
  - Feature lists
- **Kullanım:** `from src.utils.config import *`

### `__init__.py`
- **Amaç:** Utils package tanımı

---

## 🔄 Tipik İş Akışı

### **1. Veri Hazırlama**
```bash
# Ham veriden session-level veriye
python -m src.data.prepare_v3

# Output: data/v3/*.parquet
```

### **2. Feature Engineering**
```python
from src.features.engineered_features import create_session_features

df_sessions = create_session_features(df_events)
```

### **3. Model Eğitimi**
```bash
# v3.0 baseline
python -m src.models.train_kfold

# Phase 3 optimization
python -m src.models.phase3_optuna_tuning

# Phase 4 ensemble
python -m src.models.phase4_ensemble
```

### **4. Evaluation**
```bash
python -m src.evaluation.detailed_metrics_phase3
```

### **5. Analysis**
```bash
python -m src.analysis.feature_analysis
```

---

## 📋 Dosya Kullanım Durumu

| Dosya | Aktif Kullanım | Purpose |
|-------|----------------|---------|
| **models/train_kfold.py** | ✅ Production | v3.0 eğitimi |
| **models/phase*.py** | ✅ Research | Optimization denemeleri |
| **features/engineered_features.py** | ✅ Production | Feature pipeline |
| **data/prepare_v3.py** | ✅ Production | Data pipeline |
| **evaluation/*.py** | ✅ Active | Metrics & analysis |
| models/train.py | 📄 Archive | Eski, referans |
| features/advanced.py | 📄 Archive | Eski, referans |
| data/prepare.py | 📄 Archive | Eski, referans |

**Durum Açıklamaları:**
- ✅ Production: Aktif kullanımda, silme
- ✅ Research: Raporda bahsedildi, silme
- 📄 Archive: Eski ama referans için saklandı

---

## 🚫 Gereksiz Dosya YOK!

**Analiz Sonucu:** 26 dosyanın hepsi projenin bir parçası. Hiçbiri gereksiz değil.

**Neden?**
- Production files: v3.0 pipeline için gerekli
- Phase files: Tüm optimizasyon denemeleri raporda belgelendi
- Archive files: Önceki versiyonlar için referans

**Tavsiye:** Tüm dosyaları sakla. Proje tarihi ve documentation için değerli.

---

## 📦 Dependencies

```python
# Core
pandas>=1.3.0
numpy>=1.21.0

# ML Libraries
scikit-learn>=1.0.0
lightgbm>=3.3.0
xgboost>=1.5.0

# Optimization
optuna>=3.0.0

# Utilities
joblib>=1.1.0
pickle  # stdlib

# Visualization (analysis only)
matplotlib>=3.4.0
seaborn>=0.11.0
```

---

## 💻 Kod Kalitesi

**Toplam Satır:** ~5,100 satır Python kodu

**Ortalama Dosya Boyutu:** 196 satır

**En Büyük Dosya:** `phase4c_multiobjective.py` (424 satır)

**En Küçük Dosya:** `phase1_step5_validation.py` (108 satır)

**Kod Organizasyonu:**
- ✅ Modüler yapı
- ✅ Clear separation of concerns
- ✅ Reusable functions
- ✅ Consistent naming

---

## 🔍 Önemli Fonksiyonlar

### Data Preparation
- `src.data.prepare_v3.create_sessions()` - Event → Session dönüşümü

### Feature Engineering
- `src.features.engineered_features.create_session_features()` - Ana pipeline
- `src.features.engineered_features.create_engagement_features()` - Event rate, diversity

### Model Training
- `src.models.train_kfold.train_with_kfold()` - K-fold CV
- `src.models.phase3_optuna_tuning.optimize_lightgbm()` - Hyperparameter tuning

### Evaluation
- `src.evaluation.evaluate.calculate_metrics()` - Tüm metrikler
- `src.evaluation.evaluate.plot_roc_curve()` - Görselleştirme

---

## 📝 Notlar

- **Python Version:** 3.14
- **Code Style:** PEP 8 uyumlu
- **Import Convention:** Absolute imports (`from src.models import ...`)
- **Package Structure:** Her klasörde `__init__.py` mevcut

---

**Son Güncelleme:** 23 Aralık 2025  
**Toplam Kod:** ~5,100 satır Python  
**Klasörler:** 6 subdirectory  
**Dosyalar:** 26 Python files  
**Durum:** Production-ready ve fully documented
