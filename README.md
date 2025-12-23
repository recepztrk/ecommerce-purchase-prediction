# E-Commerce Purchase Prediction - Machine Learning Project

## 📊 Proje Özeti

Bu proje, e-commerce kullanıcı davranışlarından **alışveriş yapma olasılığını** tahmin eden bir makine öğrenmesi modelidir.

**Durum:** ✅ Tamamlandı  
**Final Model:** v3.0 LightGBM Baseline  
**Test AUC:** 0.7619  
**Test F1:** 0.69  
**Test Recall:** 0.98 (⭐ Çok yüksek!)

---

## 🎯 Proje Hedefleri ve Sonuç

**Hedef:** Test AUC 0.78+ (%2.4 iyileştirme)

**Sonuç:** 10 farklı optimizasyon yöntemi denendi, v3.0 baseline hala en iyi dengeli model

**Öğrenilen:** Veri kalitesi > Model karmaşıklığı

---

## 📁 Proje Yapısı

```
├── data/
│   ├── v3/                    # v3.0 baseline data (24 features)
│   ├── v3_final/              # Phase optimizations
│   └── *.parquet              # Train/val/test splits
│
├── models/
│   ├── lightgbm_v3.txt        # v3.0 baseline model
│   ├── best_extratrees.pkl    # Phase 3: Best AUC (0.7751)
│   ├── best_lightgbm.txt      # Phase 3: Optimized LightGBM
│   └── best_xgboost.pkl       # Phase 3: Optimized XGBoost
│
├── reports/
│   ├── FINAL_PROJECT_REPORT.md      # ⭐ Kapsamlı proje raporu
│   ├── PROJECT_PRESENTATION.md      # ⭐ Sunum dökümanı
│   ├── final_report_v3.md           # v3.0 detayları
│   └── phase3_detailed_metrics.csv  # Tüm model metrikleri
│
├── src/
│   ├── models/                # Model training scripts
│   ├── features/              # Feature engineering
│   ├── analysis/              # Data analysis
│   └── evaluation/            # Model evaluation
│
└── README.md                  # Bu dosya
```

---

## 🚀 Hızlı Başlangıç

### Gereksinimler

```bash
Python 3.14
pip install -r requirements.txt
```

### v3.0 Modeli Eğitme

```bash
cd "Makine Öğrenmesi Proje"
python src/models/train_kfold.py
```

### Model Değerlendirme

```bash
python src/evaluation/detailed_metrics_phase3.py
```

---

## 📊 Model Performansı

### v3.0 Baseline (Final Model)

| Metrik | Değer | Açıklama |
|--------|-------|----------|
| **Test AUC** | 0.7619 | Model sıralama yeteneği |
| **Test F1** | 0.69 | Precision-Recall dengesi |
| **Precision** | 0.65 | Pozitif tahminlerin doğruluğu |
| **Recall** | 0.98 | ⭐ 100 müşteriden 98'ini yakalıyor |
| **Train-Test Gap** | 11% | Düşük overfitting |

### Neden v3.0 En İyi?

- ✅ En yüksek F1 score (0.69)
- ✅ En yüksek Recall (0.98) - Neredeyse tüm müşterileri yakalıyor
- ✅ En dengeli metrikler
- ✅ En düşük overfitting gap (%11)
- ✅ Basit ve maintainable

---

## 🔬 Denenen Optimizasyonlar

### Başarısız Denemeler (Detaylar: `FINAL_PROJECT_REPORT.md`)

1. **v4.0** - Aggressive feature removal → AUC düştü (-2.9%)
2. **v5.0** - Additive features (68 features) → Overfitting (+3% gap)
3. **v6.0** - Stacking ensemble → Recall düştü (-13%)
4. **Phase 3** - Hyperparameter optimization → En iyi: ExtraTrees (AUC 0.7751) ama recall düşük
5. **10 Ensemble yöntemi** - Grid search, stacking, multi-objective → Hepsi başarısız
6. **v3.0 Tuning** - Optuna ile v3.0 optimize → Daha kötü sonuç!

**Sonuç:** v3.0 baseline hala en iyi dengeli model

---

## 📚 Belgeler

### Ana Raporlar

1. **[FINAL_PROJECT_REPORT.md](reports/FINAL_PROJECT_REPORT.md)** ⭐
   - Tüm denemelerin detaylı analizi
   - Her başarısızlığın teknik açıklaması
   - Öğrenilen dersler
   - Metodoloji detayları

2. **[PROJECT_PRESENTATION.md](reports/PROJECT_PRESENTATION.md)** ⭐
   - Sunum için özet format
   - Görselleştirilebilir
   - Slide yapısında

3. **[final_report_v3.md](reports/final_report_v3.md)**
   - v3.0 baseline detaylı analiz
   - Veri kalitesi metodolojisi

---

## 🎓 Öğrenilen Dersler

### 1. Data Quality > Model Complexity
v3.0'ın başarısı = Temiz veri (session merging, quality filtering)

### 2. Validation ≠ Test
Validation'da harika olan modeller test'te başarısız olabilir (overfitting)

### 3. Recall'dan Fedakarlık Yapma
v3.0'ın 0.98 recall'ı iş değeri açısından altın

### 4. Ensemble Her Zaman İyi Değil
10 yöntem denendi, hiçbiri v3.0'dan dengeli çıkmadı

### 5. Simple is Beautiful
24 feature + default parameters > 68 feature + complex ensemble

---

## 💡 Kullanım Önerileri

### Model Çıktısı: Olasılık Skorları

```python
# Model predictions (0.0 - 1.0)
predictions = model.predict(X_test)

# Müşterileri skorla ve sırala
user_scores = {
    'user_1': 0.95,  # %95 ihtimal alışveriş yapacak
    'user_2': 0.73,  # %73 ihtimal
    'user_3': 0.51,  # %51 ihtimal
    'user_4': 0.22   # %22 ihtimal
}
```

### İş Kullanımı

- **Top %10** → Kesin kampanya gönder
- **%10-30** → Orta öncelik
- **%30-50** → İndirim göster
- **%50 altı** → Hiç uğraşma

---

## 📈 Gelecek İyileştirme Önerileri

### Eğer Kaynak Bulunursa:

1. **Daha Fazla Veri** (+2-3% AUC beklenir)
   - Hedef: 10M+ session
   - En etkili iyileştirme

2. **External Features** (+1-2% AUC)
   - Ürün kategorisi detayları
   - Fiyat trendleri
   - Mevsimsellik

3. **A/B Testing Framework**
   - Gerçek kullanıcılarla test
   - Business metric tracking

---

## 🛠️ Teknik Detaylar

**Veri:**
- Train: 2.2M sessions
- Validation: 469K sessions
- Test: 541K sessions
- Features: 24

**Modeller:**
- Algorithm: LightGBM
- Features: Session-level aggregations
- Evaluation: 5-fold cross-validation
- Metrics: AUC, F1, Precision, Recall

**Araçlar:**
- Python 3.14
- Scikit-learn
- LightGBM, XGBoost
- Optuna (hyperparameter optimization)
- Pandas, NumPy

---

## 📝 Nasıl Cite Edilir

Eğer bu projeyi kullanıyorsanız, lütfen cite edin:

```
E-Commerce Purchase Prediction
Machine Learning Project
2025
```

---

## 🙏 Katkıda Bulunanlar

**Proje:** E-Commerce Purchase Prediction  
**Durum:** Tamamlandı  
**Tarih:** Aralık 2025

---

## 📞 İletişim

Sorularınız için:
- **Raporlar:** `reports/` klasörü
- **Kod:** `src/` klasörü
- **Modeller:** `models/` klasörü

---

## 📄 Lisans

Bu proje akademik/eğitim amaçlıdır.

---

**Son Güncelleme:** 2025-12-23  
**Versiyon:** v3.0 (Final)
