# Models Directory

Bu klasörde eğitilmiş modeller ve eğitim sonuçları saklanır.

## 📁 Klasör İçeriği

### **v3.0 Baseline Models (Final Model)**

#### `lightgbm_v3.txt` (237 KB) ⭐
- **Model:** LightGBM v3.0 Baseline
- **Performans:** Test AUC 0.7619, F1 0.69, Recall 0.98
- **Durum:** **PRODUCTION MODEL** - Final seçilen model
- **Kullanım:**
```python
import lightgbm as lgb
model = lgb.Booster(model_file='models/lightgbm_v3.txt')
predictions = model.predict(X_test)
```

#### `xgboost_v3.json` (5.8 MB)
- **Model:** XGBoost v3.0 (alternatif)
- **Performans:** Test AUC 0.7595
- **Kullanım:** v3.0 ensemble ve karşılaştırma için

---

### **Phase 3: Hyperparameter Optimization Models**

Phase 3'te Optuna ile optimize edilmiş modeller (25 trials):

#### `best_lightgbm.txt` (658 KB)
- **Model:** LightGBM (Mac local optimization)
- **Performans:** Test AUC 0.7566
- **Kaynak:** `src/models/phase3_optuna_tuning.py`

#### `best_lightgbm_colab.txt` (931 KB)
- **Model:** LightGBM (Google Colab optimization)
- **Performans:** Colab paralel çalışması
- **Kaynak:** Colab notebook

#### `best_xgboost.pkl` (722 KB)
- **Model:** XGBoost (Mac optimization)
- **Format:** Pickle serialized

#### `best_xgboost_colab.pkl` (14 MB)
- **Model:** XGBoost (Colab optimization)
- **Performans:** Test AUC 0.7691
- **Not:** En iyi precision (0.61)

---

### **Failed Experiments (Başarısızlık Kanıtları)**

#### `v3_lightgbm_optimized.txt` (445 KB)
- **Model:** v3.0'ın Optuna ile tuned versiyonu
- **Kaynak:** `src/models/v3_hyperparameter_tuning.py` (50 trials)
- **Sonuç:** ❌ BAŞARISIZ
- **Performans:** 
  - Val AUC: 0.8154 (iyi görünüyordu)
  - Test AUC: 0.7555 (-0.84% kötü!)
  - Test F1: 0.68 (-1.45%)
  - Test Recall: 0.85 (-13.3% çok kötü!)
- **Neden saklandı:** Overtuning'in kanıtı, raporda kullanıldı

---

### **Configuration & Metrics**

#### `ensemble_phase4_weights.json` (294 B)
- **İçerik:** Phase 4 ensemble ağırlıkları
- **Format:** JSON
```json
{
  "model_weights": {
    "extratrees": 0.0,
    "xgboost": 0.9,
    "lightgbm": 0.1
  }
}
```

#### `version_comparison_v3.csv` (291 B)
- **İçerik:** v1.0, v2.0, v3.0 karşılaştırması
- **Kolonlar:** Version, Test_AUC, Val_AUC
- **Örnek:**
```csv
Version,Test_AUC,Val_AUC
v3.0 LightGBM,0.7622,0.8004
```

#### `training_log.txt` (2.5 KB)
- **İçerik:** Genel training log dosyası
- **Kullanım:** Debug ve analiz için

---

## 📊 Model Performans Özeti

| Model | Test AUC | F1 | Recall | Gap | Durum |
|-------|----------|-----|--------|-----|-------|
| **v3.0 LightGBM** | **0.7619** | **0.69** | **0.98** ⭐ | **11%** | ✅ Production |
| XGBoost (Colab) | 0.7691 | 0.64 | 0.67 | 13.6% | Phase 3 |
| LightGBM (Colab) | 0.7566 | 0.68 | 0.85 | 13.7% | Phase 3 |
| v3.0 Tuned | 0.7555 | 0.68 | 0.85 | 13% | ❌ Başarısız |

**Final Karar:** v3.0 Baseline hala en dengeli model (5 metrikten 4'ünde en iyi)

---

## 🚫 GitHub'a Yüklenmeyen Dosyalar

Büyük model dosyaları `.gitignore` ile filtrelendi:
- `*.pkl` (pickle dosyaları)
- `*.txt` (LightGBM modelleri)
- `*.json` büyük XGBoost modelleri

**Sadece CSV ve JSON config dosyaları GitHub'da**

---

## 🔧 Model Kullanımı

### Production Model (v3.0) Yükleme

```python
import lightgbm as lgb
import pandas as pd

# Model yükle
model = lgb.Booster(model_file='models/lightgbm_v3.txt')

# Tahmin yap
X_test = pd.read_parquet('data/v3/test_sessions_v3.parquet')
predictions = model.predict(X_test.drop(['target', 'user_session', 'user_id'], axis=1))

# Olasılık skorları
print(f"Prediction scores: {predictions[:5]}")
```

### Model Yeniden Eğitme

```bash
# v3.0 baseline
python -m src.models.train_kfold

# Phase 3 optimization
python -m src.models.phase3_optuna_tuning
```

---

## 📝 Notlar

- **Final Model:** `lightgbm_v3.txt` - Production'da kullanılabilir
- **Dosya Boyutu:** ~89 MB (önceki temizlikten sonra)
- **Temizlik:** 66MB training_results_v3.pkl silindi (redundant)
- **Detaylı Rapor:** `../reports/FINAL_PROJECT_REPORT.md`

---

**Son Güncelleme:** 23 Aralık 2025  
**Production Model:** v3.0 LightGBM Baseline
