# Data Directory

Bu klasörde işlenmiş session-level veriler bulunur.

## 📁 Klasör Yapısı

### `v3/` - v3.0 Baseline Verisi (Final Model)
v3.0 baseline modelinin kullandığı data (24 features, session-level):
- `train_sessions_v3.parquet` - Eğitim verisi (2.2M sessions)
- `val_sessions_v3.parquet` - Validasyon verisi (469K sessions)
- `test_sessions_v3.parquet` - Test verisi (541K sessions)
- `train_features_v3.parquet` - Detaylı feature set
- `val_features_v3.parquet`
- `test_features_v3.parquet`

**Kullanım:** Final model (v3.0 LightGBM) bu veriyi kullanıyor.

### `v3_final/` - Phase Optimization Verisi
Phase 1-4 optimizasyonlarında kullanılan enhanced data:
- `train_sessions_final.parquet` - Enhanced training data
- `val_sessions_final.parquet` - Enhanced validation data
- `test_sessions_final.parquet` - Enhanced test data

**Not:** Phase çalışmaları için referans veri.

## 📊 Veri Detayları

**Veri Raporu:** `../final_reports/PROCESSED_DATASET_REPORT.md`

**Özet:**
- Veri Seviyesi: Session-level (her satır bir kullanıcı oturumu)
- Feature Sayısı: 24 (v3.0)
- Kayıt Sayısı: ~3.2M sessions (toplam)
- Format: Apache Parquet (sıkıştırılmış, hızlı)

## 🔧 Veriyi Kullanma

```python
import pandas as pd

# v3.0 baseline veri
train = pd.read_parquet('data/v3/train_sessions_v3.parquet')
val = pd.read_parquet('data/v3/val_sessions_v3.parquet')
test = pd.read_parquet('data/v3/test_sessions_v3.parquet')

print(f"Train shape: {train.shape}")
print(f"Features: {train.columns.tolist()}")
```

## ℹ️ Notlar

- Büyük dosyalar olduğu için GitHub'a yüklenmemiştir (.gitignore)
- Orijinal ham veri: `../archive/` klasöründe
- Veri transformasyon detayları: `../final_reports/` klasöründeki raporlarda
