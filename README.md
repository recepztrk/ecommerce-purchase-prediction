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

### Model Performansı

| Model | Val ROC-AUC | Test ROC-AUC | Test PR-AUC | Test F1 |
|-------|-------------|--------------|-------------|---------|
| Naive Baseline | 0.5000 | 0.5000 | 0.4273 | 0.0000 |
| Logistic Regression | 0.6251 | 0.5833 | 0.4938 | 0.5548 |
| **LightGBM** | **0.6492** | **0.5936** | **0.4838** | **0.6272*** |

*Optimal threshold (0.40) ile

### En İyi Model: LightGBM

**Test Set Performansı (threshold=0.40):**
- **ROC-AUC:** 0.5936
- **PR-AUC:** 0.4838
- **F1 Score:** 0.6272
- **Precision:** 0.4577
- **Recall:** 0.9960

**Top 5 Önemli Özellikler:**
1. `ts_day_mean` - Oturum günü
2. `events_per_minute` - Oturum yoğunluğu
3. `product_diversity` - Ürün çeşitliliği
4. `ts_month_mean` - Oturum ayı
5. `session_duration_seconds` - Oturum süresi

### Grafikler

Detaylı grafikler ve analizler için `reports/` klasörüne bakın:
- `roc_pr_curves.png` - ROC ve PR eğrileri
- `confusion_matrices.png` - Confusion matrix'ler
- `feature_importance.png` - Özellik önem sıralaması
- `report.md` - Detaylı teknik rapor

## Geliştirici

Makine Öğrenmesi Projesi - 2025

