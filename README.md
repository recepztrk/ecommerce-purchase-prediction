# E-Commerce Satın Alma Tahmini Projesi

## Proje Özeti

E-commerce platformunda kullanıcı oturumlarının satın alma ile sonuçlanıp sonuçlanmayacağını tahmin eden bir makine öğrenmesi projesi.

**Hedef:** Session-level ikili sınıflandırma (oturum satın alma ile sonuçlanır mı?)

**Veri:** ~11.5M event (cart + purchase), Parquet format

## Kurulum

```bash
# Virtual environment oluştur
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Bağımlılıkları kur
pip install -r requirements.txt
```

## Veri Yerleşimi

```
archive/
  ├── train.parquet
  ├── val.parquet
  └── test.parquet
```

## Kullanım

```bash
# Veri hazırlama ve EDA
python -m src.data.prepare

# Feature engineering
python -m src.features.build

# Model eğitimi
python -m src.models.train

# Değerlendirme
python -m src.evaluation.evaluate
```

## Proje Yapısı

```
.
├── README.md
├── requirements.txt
├── archive/              # Ham veri
├── data/                 # İşlenmiş veri
├── models/               # Kaydedilmiş modeller
├── reports/              # Raporlar ve grafikler
├── notebooks/            # Jupyter notebook'lar
└── src/
    ├── data/            # Veri okuma ve hazırlama
    ├── features/        # Feature engineering
    ├── models/          # Model tanımları
    ├── evaluation/      # Metrik ve değerlendirme
    └── utils/           # Yardımcı fonksiyonlar
```

## Hedef Tanımı ve Leakage Önlemi

**Orijinal Veri:** Event-level (her satır bir event: cart veya purchase)

**Dönüşüm:** Session-level aggregation
- Target = 1: Oturumda en az bir purchase var
- Target = 0: Oturumda sadece cart event'leri var

**Leakage Kontrolü:**
- `event_type` kolonu feature olarak kullanılmıyor
- Session içi davranış patternleri, zaman özellikleri, ürün özellikleri kullanılıyor
- Train/val/test split user_session bazlı (aynı session farklı split'lere düşmüyor)

## Sonuçlar

### 🚀 v2.0 (Current - Improved)

| Model | Val ROC-AUC | Test ROC-AUC | Improvement vs v1.0 |
|-------|-------------|--------------|---------------------|
| **LightGBM v2** | **0.6596** | **0.6107** | **+2.88%** ⭐ |
| XGBoost | 0.6578 | 0.6098 | +2.73% |
| Ensemble | 0.6593 | 0.6107 | +2.88% |

**Key Improvements in v2.0:**
- ✅ 42 → 59 features (+17 advanced features)
- ✅ ROC-AUC: 0.5936 → 0.6107 (+2.88%)
- ✅ 3 models (LightGBM + XGBoost + Ensemble)
- ✅ Optimized hyperparameters

### 📊 v1.0 (Baseline)

| Model | Val ROC-AUC | Test ROC-AUC | Test PR-AUC | Test F1 |
|-------|-------------|--------------|-------------|---------|
| Naive Baseline | 0.5000 | 0.5000 | 0.4273 | 0.0000 |
| Logistic Regression | 0.6251 | 0.5833 | 0.4938 | 0.5548 |
| LightGBM v1 | 0.6492 | 0.5936 | 0.4838 | 0.6272 |

### 🆕 New Features (v2.0)

**Sequence Features:**
- Event timing patterns, acceleration metrics

**Price Trajectory:**
- Price trends, volatility, ascending patterns

**Behavioral Scores:**
- Focus score, exploration score, decisiveness score

**Temporal Patterns:**
- Hour consistency, time gap statistics

### Grafikler ve Raporlar

Detaylı grafikler ve analizler için `reports/` klasörüne bakın:

**v2.0 (Current):**
- `model_comparison_v2.png` - Model karşılaştırma eğrileri
- `improvement_report_v2.md` - Detaylı iyileştirme raporu

**v1.0 (Baseline):**
- `roc_pr_curves.png` - ROC ve PR eğrileri
- `confusion_matrices.png` - Confusion matrix'ler
- `feature_importance.png` - Özellik önem sıralaması
- `report.md` - Detaylı teknik rapor

## Geliştirici

Makine Öğrenmesi Projesi - 2025

