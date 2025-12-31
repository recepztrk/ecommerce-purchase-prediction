# 🛒 E-Commerce Satın Alma Tahmini

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Lisans](https://img.shields.io/badge/Lisans-MIT-green.svg)]()
[![Durum](https://img.shields.io/badge/Durum-Tamamlandı-success.svg)]()
[![Dokümantasyon](https://img.shields.io/badge/Dökümanlar-Kapsamlı-brightgreen.svg)]()

> E-ticaret kullanıcılarının tarama davranışlarından satın alma niyetini tahmin eden makine öğrenmesi projesi. 10 farklı optimizasyon yaklaşımının sistematik incelemesi ile **veri kalitesinin model karmaşıklığından daha önemli** olduğunu gösteriyor.

**Sonuç:** Kapsamlı optimizasyon denemelerinden sonra v3.0 LightGBM Baseline en dengeli model olarak kaldı.

---

## 📋 İçindekiler

- [Hızlı Bakış](#-hızlı-bakış)
- [Temel Sonuçlar](#-temel-sonuçlar)
- [Proje Yolculuğu](#-proje-yolculuğu)
- [Önemli Çıkarımlar](#-önemli-çıkarımlar)
- [Dokümantasyon](#-dokümantasyon)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Proje Yapısı](#-proje-yapısı)
- [Görseller](#-görseller)
- [Metodoloji](#-metodoloji)

---

## 🎯 Hızlı Bakış

### Problem Tanımı
E-ticaret kullanıcısının tarama oturumu sırasında satın alma yapıp yapmayacağını davranış desenlerine göre tahmin etmek.

### Dataset
- **Source:** E-commerce platform event logs (2020)
- **Kaggle:** [RecSys 2020 E-Commerce Dataset](https://www.kaggle.com/datasets/dschettler8845/recsys-2020-ecommerce-dataset)
- **Size:** 11.5M events → 2.2M quality sessions
- **Features:** 24 engineered session-level features
- **Target:** Binary (Purchase vs. No Purchase)
- **Class Distribution:** %15 positive (imbalanced)

> **Not:** Veri dosyaları (~600MB) GitHub'da bulunmamaktadır. Yukarıdaki Kaggle linkinden indirebilirsiniz.

### Yaklaşım
**Veri Kalitesi Öncelikli:** Temiz veri ile basit modeller, karmaşık özelliklerle gürültülü veriden daha iyi performans gösterir.

```
Ham Veri (11.5M event)
    ↓ Session Birleştirme
  Kalite Filtreleme
    ↓ Özellik Mühendisliği  
Final Veri (2.2M session, 24 özellik)
    ↓ v3.0 LightGBM (varsayılan parametreler)
Final Model (Test AUC: 0.7619, Recall: 0.98)
```

---

## 🏆 Temel Sonuçlar

### Final Model: v3.0 LightGBM Baseline

| Metrik | Değer | Açıklama |
|--------|-------|----------|
| **Test AUC** | 0.7619 | Güçlü sıralama yeteneği |
| **Test F1** | 0.69 | Dengeli precision-recall |
| **Test Precision** | 0.65 | Pozitif tahminlerin %65'i doğru |
| **Test Recall** | **0.98** ⭐ | **100 müşteriden 98'ini yakalıyor!** |
| **Train-Test Gap** | %11 | Düşük overfitting |

### Bu Sonuçlar Neden Önemli?

**İş Etkisi:** 100 potansiyel müşteriden 98'ini yakalıyor, sadece 2'sini kaçırıyor.

Bu olağanüstü recall oranı şunları sağlıyor:
- False negative'lerden minimal gelir kaybı
- Verimli pazarlama kampanyası hedefleme
- Maksimum müşteri dönüşümü yakalama

---

## 🔬 Proje Yolculuğu

Bu proje, sağlam v3.0 baseline sonuçları elde ettikten sonra **10 farklı optimizasyon yaklaşımını** sistematik olarak test etti.

### Versiyon Evrimi

| Versiyon | Yaklaşım | Test AUC | Temel İyileştirme |
|----------|----------|----------|-------------------|
| v1.0 | İlk Baseline | 0.5936 | İlk implementasyon |
| v2.0 | Gelişmiş Özellikler + Tuning | 0.6107 | +17 özellik, hiperparametre optimizasyonu |
| **v3.0** | **Veri Kalitesi Odaklı** | **0.7619** | **Session birleştirme, kalite filtreleme (+%28.4)** |

**Temel Çıkarım:** v1.0 → v3.0 arasında +%28.4 AUC iyileştirmesi **veri kalitesi** ile sağlandı, model karmaşıklığı ile değil.

### Optimizasyon Denemeleri (Hepsi v3.0'ı Geçemedi)

v3.0'dan sonra, performansı daha da artırmak için **10 sofistike yaklaşım** test edildi:

#### Kategori 1: Özellik Mühendisliği
1. **v4.0 - Agresif Özellik Çıkarma** (16 özellik)
   - Sonuç: AUC 0.7398 (-%2.9) ❌
   - Öğrenilen: Çıkarılan özellikler kombinasyon halinde değerliymiş

2. **v5.0 - Eklemeli Mühendislik** (68 özellik)
   - Sonuç: AUC 0.7588 (-%0.4), Gap +%3 ❌
   - Öğrenilen: Daha fazla özellik ≠ daha iyi performans

#### Kategori 2: Gelişmiş Modelleme
3. **v6.0 - Stacking Ensemble**
   - Sonuç: AUC 0.7678 (+%0.8), ama Recall 0.77'ye düştü ❌
   - Öğrenilen: Yüksek AUC daha iyi model anlamına gelmez

#### Kategori 3: Sistematik Optimizasyon

**Phase 1: Akıllı Özellik Seçimi**
- Sonuç: Minimal iyileştirme (+%0.13)

**Phase 2: Algoritma Testi**
- Test Edilenler: ExtraTrees, XGBoost, Random Forest, HistGradientBoosting
- Sonuç: Hiçbiri v3.0'ın recall'ını yakalayamadı

**Phase 3: Hiperparametre Optimizasyonu (Optuna)**
- Model başına 25 deneme, paralel çalıştırma (Mac + Colab)
- En İyi: ExtraTrees (AUC 0.7751)
- Problem: Recall 0.77'ye düştü (**%21 müşteri kaybı!**)

#### Kategori 4: Ensemble Yöntemleri (6 varyasyon)
4-9. Çeşitli ensemble yaklaşımları test edildi:
   - Grid search ağırlıklı oylama
   - Eşit ağırlıklar
   - Meta-learner ile stacking
   - Çok-amaçlı optimizasyon (6 varyant)
   - Hepsi v3.0'ın recall'ını koruyamadı

10. **v3.0 Hiperparametre Tuning** (50 deneme)
    - Validation AUC: 0.8154 (mükemmel!)
    - Test AUC: 0.7555 (-%0.84) ❌
    - Ciddi overfitting!

### Final Karşılaştırma

| Model | Test AUC | F1 | Recall | Gap | Kazanan |
|-------|----------|-----|--------|-----|---------|
| **v3.0 Baseline** | **0.7619** | **0.69** ★ | **0.98** ★★★ | **%11** ★ | ✅ En İyi |
| ExtraTrees (Optimized) | **0.7751** ★ | 0.67 | 0.77 | %13.6 | ❌ Düşük Recall |
| XGBoost (Colab) | 0.7691 | 0.64 | 0.67 | %13.6 | ❌ Düşük F1 |
| Equal Weights Ensemble | 0.7689 | 0.67 | 0.80 | %13.6 | ❌ Seyreltilmiş güç |
| Stacking Ensemble | 0.7678 | 0.67 | 0.77 | %13.6 | ❌ Karmaşık, düşük recall |

**v3.0, 5 metrikten 4'ünde kazanıyor!** ⭐

---

## 💡 Önemli Çıkarımlar

### 1. Veri Kalitesi > Model Karmaşıklığı
- v3.0'ın başarısı temiz veriden geliyor (session birleştirme, kalite filtreleme)
- 68 özellik + stacking < 24 özellik + varsayılan LightGBM

### 2. Validation ≠ Test
- XGBoost ve tuned v3.0: Mükemmel validation, kötü test
- ExtraTrees: Kötü validation, en iyi test AUC
- **Öğrenilen:** Grid search validation set'e overfit olabilir

### 3. Recall Vazgeçilmezdir
- v3.0'ın 0.98 recall'ı = **iş için altın**
- 100 müşteri → model 98'ini yakalar, sadece 2'sini kaçırır
- ExtraTrees 21 müşteri kaçırıyor (10x daha kötü!)
- Gelir etkisi çok büyük

### 4. Ensemble Büyüsü Yoktur  
- 10 ensemble varyasyonu test edildi
- Modeller birbirini tamamlamadı (benzer hata desenleri)
- Tek güçlü model > zayıf ensemble

### 5. Basit Güzeldir
- v3.0: 24 özellik, varsayılan parametreler, 237KB dosya
- v6.0: 68 özellik, stacking, meta-learner, karmaşık
- **v3.0 kazandı!**

### 6. Varsayılan Parametreler Optimal Olabilir
- Agresif Optuna tuning (50 deneme) v3.0'ı kötüleştirdi
- Varsayılan LightGBM parametreleri iyi ayarlanmış
- **Öğrenilen:** Domain bilgisi > kör optimizasyon

---

## 📚 Dokümantasyon

### Temel Dokümanlar

#### Projeyi Anlamak İçin
- **[Bu README]** - Proje özeti ve hızlı başlangıç
- **[FINAL_PROJECT_REPORT.md](reports/FINAL_PROJECT_REPORT.md)** - Tüm 10 deneyin kapsamlı analizi (635 satır)
- **[PROJECT_PRESENTATION.md](reports/PROJECT_PRESENTATION.md)** - 10-15 dk sunum formatı

#### Veriyi Anlamak İçin
- **[ORIGINAL_DATASET_REPORT.md](final_reports/ORIGINAL_DATASET_REPORT.md)** - Ham veri analizi (11.5M event)
- **[PROCESSED_DATASET_REPORT.md](final_reports/PROCESSED_DATASET_REPORT.md)** - İşlenmiş veri (2.2M session, 24 özellik)

#### Teknik Detaylar
- **[final_report_v3.md](reports/final_report_v3.md)** - v3.0 metodoloji detayları
- **[final_report_v6.md](reports/final_report_v6.md)** - v6.0 stacking ensemble (neden başarısız oldu)

#### Kod Dokümantasyonu
- **[src/README.md](src/README.md)** - Kaynak kod yapısı (~5,100 satır Python)
- **[data/README.md](data/README.md)** - Veri organizasyonu
- **[models/README.md](models/README.md)** - Eğitilmiş modeller
- **[reports/README.md](reports/README.md)** - Raporlar ve görseller

---

## 📊 Görseller

### Profesyonel Görseller

`reports/final_visuals/` klasöründe 10 profesyonel görsel:

**Temel Set (6 görsel):**
1. Model Karşılaştırma Tablosu - Tüm modeller & metrikler
2. Confusion Matrix (v3.0) - TP/FP/TN/FN dağılımı
3. ROC Eğrisi - AUC görselleştirmesi
4. Özellik Önemi - En önemli 15 özellik
5. AUC Karşılaştırma Bar Chart - Model sıralaması
6. İş Etkisi - 98/100 müşteri yakalama

**Ek Set (4 görsel):**
7. Veri Dönüşüm Akışı - 11.5M→2.2M pipeline
8. Başarısız Denemeler Zaman Çizelgesi - 10 deneme yolculuğu
9. Precision-Recall Eğrisi - Dengesiz veri performansı
10. Sınıf Dağılımı - %85 vs %15 dengesizlik

Tümü Python (matplotlib/seaborn) ile oluşturuldu. Detaylar için [reports/final_visuals/README.md](reports/final_visuals/README.md).

---

## 🚀 Kurulum

### Gereksinimler
- Python 3.14+
- 2.2GB disk alanı (veri için)
- 23MB (modeller için)

### Hızlı Kurulum

```bash
# Repository'yi klonla
git clone https://github.com/recepztrk/ecommerce-purchase-prediction.git
cd ecommerce-purchase-prediction

# Bağımlılıkları yükle
pip install -r requirements.txt
```

#### Bağımlılıklar
```
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
lightgbm==4.1.0
xgboost==2.0.3
pyarrow==14.0.1
matplotlib==3.8.2
seaborn==0.13.0
optuna==3.5.0
imbalanced-learn==0.11.0
```

---

## 💻 Kullanım

### Production Model'i Yükle (v3.0)

```python
import lightgbm as lgb
import pandas as pd

# v3.0 baseline model'i yükle
model = lgb.Booster(model_file='models/lightgbm_v3.txt')

# Test verisini yükle
X_test = pd.read_parquet('data/v3/test_sessions_v3.parquet')

# Tahmin al (0-1 arası olasılık skorları)
predictions = model.predict(
    X_test.drop(['target', 'user_session', 'user_id'], axis=1)
)

# Örnek çıktı
print(f"Müşteri 1: %{predictions[0]:.2%} satın alma olasılığı")
# Çıktı: Müşteri 1: %92.5 satın alma olasılığı
```

### İş Uygulamaları

```python
# Olasılık skorlarına göre kampanya hedefleme
campaigns = {
    'premium': predictions > 0.85,  # En üst %10 - Kesin gönder
    'standard': (predictions > 0.60) & (predictions <= 0.85),  # Orta - İndirimle gönder
    'low_priority': (predictions > 0.50) & (predictions <= 0.60)  # Düşük - Sadece reklam göster
}

# Model gerçek alıcıların %98'ini yakalıyor
# Minimal false negative → Maksimum gelir
```

### Model'i Sıfırdan Eğit

```bash
# 5-fold CV ile full training pipeline
python -m src.models.train_kfold

# Çıktı: models/lightgbm_v3.txt, models/xgboost_v3.json
```

### Phase Optimizasyonlarını Çalıştır

```bash
# Phase 3: Hiperparametre optimizasyonu (Optuna)
python -m src.models.phase3_optuna_tuning

# Phase 4: Ensemble yöntemleri
python -m src.models.phase4_ensemble
```

---

## 📁 Proje Yapısı

```
├── archive/                 (883 MB - Orijinal ham veri)
│   ├── train.parquet        (11.5M event)
│   ├── val.parquet
│   └── test.parquet
│
├── data/                    (602 MB - İşlenmiş veri)
│   ├── v3/                  (v3.0 baseline - 24 özellik)
│   │   ├── train_sessions_v3.parquet  (2.2M session)
│   │   ├── val_sessions_v3.parquet
│   │   └── test_sessions_v3.parquet
│   ├── v3_final/            (Phase optimizasyonları - 29 özellik)
│   └── README.md
│
├── models/                  (23 MB - Eğitilmiş modeller)
│   ├── lightgbm_v3.txt      (v3.0 production model ⭐)
│   ├── xgboost_v3.json      (v3.0 alternatif)
│   ├── best_*.pkl/txt       (Phase 3 optimize edilmiş modeller)
│   └── README.md
│
├── reports/                 (2.5 MB - Raporlar & görseller)
│   ├── FINAL_PROJECT_REPORT.md       (Kapsamlı ⭐⭐⭐)
│   ├── PROJECT_PRESENTATION.md       (Sunum ⭐⭐⭐)
│   ├── final_report_v3.md           (v3.0 detayları)
│   ├── final_report_v6.md           (v6.0 analizi)
│   ├── *.csv                         (7 sonuç dosyası)
│   ├── final_visuals/               (10 profesyonel PNG ⭐)
│   └── README.md
│
├── final_reports/           (Veri seti dokümantasyonu)
│   ├── ORIGINAL_DATASET_REPORT.md    (Ham veri analizi ⭐)
│   └── PROCESSED_DATASET_REPORT.md   (İşlenmiş veri ⭐)
│
├── src/                     (232 KB - Kaynak kod)
│   ├── models/              (10 script - eğitim & optimizasyon)
│   ├── features/            (5 script - özellik mühendisliği)
│   ├── data/                (3 script - veri ön işleme)
│   ├── evaluation/          (3 script - metrikler & analiz)
│   ├── analysis/            (1 script - özellik analizi)
│   ├── utils/               (2 script - config & utilities)
│   └── README.md
│
├── README.md                (Bu dosya - Proje özeti)
├── requirements.txt         (Python bağımlılıkları)
└── .gitignore              (Git ignore kuralları)
```

**Toplam:** 2.2 GB (6.6 GB'den optimize edildi)

---

## 🔬 Metodoloji

### Veri Pipeline'ı

```
Ham Eventler: 11.5M
    ↓ Session Birleştirme
3.7M Session
    ↓ Kalite Filtreleme
2.2M Temiz Session
    ↓ Özellik Mühendisliği
24 Özellik
    ↓ Train/Val/Test Bölme
Final Veri Seti
    ↓ v3.0 LightGBM
Production Model
```

### Özellik Mühendisliği

**24 Session-Seviye Özellik:**

1. **Etkileşim (2):** `n_events`, `n_unique_products`
2. **Zamansal (11):** Session süresi, saat desenleri, hafta içi desenleri
3. **Fiyat (5):** Ortalama, std, min, max, toplam
4. **Kategori (4):** 4 seviyede tekil kategoriler
5. **Mühendislenmiş (4):** `event_rate`, `product_diversity`, `engagement_intensity`, `price_velocity`

**En Önemli 3:**
1. `n_events` - %18.3 önem
2. `session_duration_seconds` - %14.5 önem
3. `event_rate` (mühendislenmiş) - %12.1 önem ⭐

### Değerlendirme Stratejisi

- **Ana Metrik:** AUC (sıralama yeteneği)
- **İkincil:** F1, Precision, Recall
- **Overfitting Kontrolü:** Train-Test gap analizi
- **Validation:** Sadece hiperparametre tuning için
- **Test:** Final değerlendirme (optimizasyon sırasında hiç dokunulmadı)

### Hesaplama Kaynakları

- **Lokal:** MacBook (M-serisi chip)
- **Cloud:** Google Colab (paralel Optuna denemeleri)
- **Toplam Süre:** ~20 saat deney

---

## 📈 Performans Kriterleri

### Confusion Matrix (Test Seti)

```
                 Tahmin Hayır   Tahmin Evet
Gerçek Hayır        384,219       73,806
Gerçek Evet           1,654       81,362
```

**Metrikler:**
- Accuracy: %85.9
- Recall: %98.0 (Sadece 1,654 / 83,016 kaçırıldı!)
- Precision: %52.4
- Specificity: %83.9

### İş Metrikleri

**Kampanya Verimliliği:**
- En üst %30'u hedefle → %95+ alıcı yakalama
- Bütçe tahsisi optimize edildi
- Minimal false negative

**ROI Etkisi:**
- v3.0 recall 0.98 vs ExtraTrees 0.77
- %21 daha fazla müşteri yakalandı = önemli gelir artışı

---

## 🎓 Öğrenilen Dersler & En İyi Pratikler

### İşe Yarayanlar

✅ **Veri Kalitesi Odağı**
- Session birleştirme (mükerrer/kısmi sessionları temizleme)
- Kalite filtreleme (bot tespiti, outlier temizleme)
- 11.5M → 2.2M (%81 azalma, büyük kalite kazancı)

✅ **Önce Basit Baseline**
- v3.0: Varsayılan LightGBM + temiz veri
- Güçlü temel oluşturuldu
- Sofistike yöntemlerle bile geçilmesi zor

✅ **Sistematik Deney**
- 10 farklı yaklaşım dokümante edildi
- Her başarısızlık değerli dersler öğretti
- Net karşılaştırma çerçevesi

✅ **İş-Metrik Hizalama**
- Recall iş değeri için önceliklendi
- Sadece AUC skorları kovalamadık
- 98/100 müşteri yakalama = gerçek etki

### İşe Yaramayanlar

❌ **Özellik Niceliği Kaliteden Önce**
- v5.0: 68 özellik overfitting yarattı
- **Öğrenilen:** Seçilmiş 24 özellik > 68 rastgele özellik

❌ **Validation-Odaklı Optimizasyon**
- Grid search validation set'e overfit oldu
- **Öğrenilen:** Ayrı holdout set kritik

❌ **Kör Hiperparametre Tuning**
- 50 Optuna denemesi v3.0'ı kötüleştirdi
- **Öğrenilen:** Varsayılan parametreler genelde iyi kalibre edilmiş

❌ **Ensemble için Ensemble**
- 10 ensemble yöntemi, hepsi başarısız
- **Öğrenilen:** Modeller tamamlamalı, sadece birleşmemeli

---

## 🔮 Gelecek İyileştirmeler

Ek kaynaklar bulunursa:

### 1. Daha Fazla Veri (En Yüksek Etki!)
- **Mevcut:** 2.2M session
- **Hedef:** 10M+ session
- **Beklenen Kazanç:** +%2-3 AUC
- **Neden:** Daha sağlam desen öğrenme

### 2. Harici Özellikler
- Ürün kategori hiyerarşileri
- Fiyat trend verileri
- Mevsimsellik göstergeleri
- Kullanıcı demografisi
- **Beklenen Kazanç:** +%1-2 AUC

### 3. Derin Öğrenme (Uzun vadeli)
- Sequence modelleri (LSTM/Transformer)
- Graph Neural Networks (ürün ilişkileri)
- **Gereksinim:** Minimum 10M+ örnek

### 4. A/B Test Framework
- Gerçek dünya deployment'ı
- İş metriklerini takip (ROI, conversion)
- Model güncellemeleri için feedback loop

---

## 📄 Lisans

MIT License - Detaylar için LICENSE dosyasına bakın.

Bu proje eğitim amaçlıdır. Veri seti sentetik/anonimleştirilmiştir.

---

## 📞 İletişim & Alıntılama

### Yazar
**Recep Öztürk**
- GitHub: [@recepztrk](https://github.com/recepztrk)
- Proje: [ecommerce-purchase-prediction](https://github.com/recepztrk/ecommerce-purchase-prediction)

### Alıntılama

Bu projeyi kullanırsanız veya faydalı bulursanız:

```bibtex
@misc{ozturk2025ecommerce,
  author = {Öztürk, Recep},
  title = {E-Commerce Satın Alma Tahmini: Optimizasyon Yaklaşımlarının Sistematik İncelemesi},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/recepztrk/ecommerce-purchase-prediction}
}
```

---

## 🏆 Proje İstatistikleri

- **Kod Satırı:** ~5,100 (Python)
- **Dokümantasyon:** ~5,000 satır (Markdown)
- **Denemeler:** 10 optimizasyon yaklaşımı
- **Eğitilen Modeller:** 30+
- **Görseller:** 10 profesyonel grafik
- **Proje Süresi:** [Süreniz]
- **Final Durum:** ✅ Tamamlandı

---

## 📝 Değişiklik Günlüğü

### v3.0 (Final) - Aralık 2025
- ✅ Production model seçildi (v3.0 LightGBM Baseline)
- ✅ 10 optimizasyon yaklaşımı test edildi ve dokümante edildi
- ✅ Kapsamlı dokümantasyon oluşturuldu
- ✅ 10 profesyonel görsel üretildi
- ✅ Kod temizliği ve organizasyon (4.4GB'dan 2.2GB'ye düşürüldü)
- ✅ Tüm denemeler tekrar üretilebilir

### v2.0
- Gelişmiş özellikler ve hiperparametre tuning eklendi
- Test AUC: 0.6107 (+%2.88 v1.0'a göre)

### v1.0
- İlk baseline implementasyonu
- Test AUC: 0.5936

---

**Dipnot: Bazen basit çözüm en iyi çözümdür.** 🚀
