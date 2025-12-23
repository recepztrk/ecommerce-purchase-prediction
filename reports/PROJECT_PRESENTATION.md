# E-Commerce Purchase Prediction
## Proje Sunumu

---

## 📌 Proje Özeti

**Hedef:** E-commerce kullanıcı davranışlarından alışveriş yapma olasılığını tahmin etmek

**Başlangıç:** v3.0 baseline (Test AUC 0.7619)

**Hedef:** Test AUC 0.78+ (%2.4 iyileştirme)

**Sonuç:** v3.0 hala en iyi model

**Denenen Yöntemler:** 10 farklı optimizasyon yaklaşımı

---

## 🎯 v3.0 Baseline Performansı

| Metrik | Değer | Açıklama |
|--------|-------|----------|
| Test AUC | 0.7619 | Sıralama yeteneği |
| Test F1 | 0.69 | Precision-Recall dengesi |
| Precision | 0.65 | Doğruluk |
| **Recall** | **0.98** | **⭐ Müşteri yakalama** |
| Gap | 11% | Düşük overfitting |

**Güçlü Yön:** %98 recall - 100 müşteriden sadece 2'sini kaçırıyor!

---

## 🔬 Denenen Optimizasyonlar

### 1. Feature Engineering

**v4.0 - Aggressive Removal (16 features)**
- Sonuç: AUC 0.7398 (-2.9%) ❌
- Neden başarısız: Bilgi kaybı

**v5.0 - Additive (68 features)**
- Sonuç: AUC 0.7588 (-0.4%) ❌
- Neden başarısız: Overfitting (+3% gap)

---

### 2. Algorithm Testing

| Algorithm | AUC | F1 | Recall | Sonuç |
|-----------|-----|-----|--------|-------|
| ExtraTrees | 0.7644 | 0.67 | 0.77 | ❌ Recall düşük |
| XGBoost | 0.7623 | 0.68 | 0.84 | ❌ Recall düşük |
| LightGBM | 0.7629 | 0.67 | 0.83 | ❌ Recall düşük |

**Hiçbiri v3.0'ın recall'ını (0.98) yakalayamadı**

---

### 3. Hyperparameter Optimization (Optuna)

**ExtraTrees (25 trials):**
- Test AUC: 0.7751 (+1.73%) ✅
- F1: 0.67 (-2.9%) ❌
- Recall: 0.77 (-21%!) ❌
- Gap: 13.6% (+2.6%) ❌

**Sonuç:** AUC arttı ama dengesi bozuldu

---

### 4. Ensemble Methods (10 Yöntem!)

**Denenen Yöntemler:**
1. Grid Search (AUC optimization)
2. Equal Weights  
3. Stacking (Meta-learner)
4. Multi-objective: AUC only
5. Multi-objective: F1 only
6. Multi-objective: AUC + F1
7. Multi-objective: AUC + F1 + Precision
8. Multi-objective: AUC + F1 + Recall
9. Multi-objective: Composite
10. v3.0 Hyperparameter Tuning

**Sonuç:** HEPSİ başarısız! ❌

---

### 5. En İyi Ensemble Sonucu

**AUC+F1+Precision Optimization:**
- Test AUC: 0.7702 (+1.09%)
- F1: 0.66 (-4.3%)
- Recall: 0.72 (-26%!) ❌

**v3.0 hala daha dengeli**

---

## 📊 Karşılaştırma Tablosu

| Model | AUC | F1 | Recall | Gap |
|-------|-----|-----|--------|-----|
| **v3.0** | 0.7619 | **0.69** | **0.98** | **11%** |
| ExtraTrees | **0.7751** | 0.67 | 0.77 | 13.6% |
| Ensemble | 0.7702 | 0.66 | 0.72 | 13.6% |
| v3.0 Tuned | 0.7555 | 0.68 | 0.85 | 13% |

**v3.0: 5 metrikten 3'ünde #1**

---

## 🎓 Öğrenilen Dersler

### 1. Data Quality > Everything
v3.0'ın başarısı = Temiz veri (session merging, quality filtering)

### 2. Validation ≠ Test
Validation'da harika → Test'te kötü (overfitting!)

### 3. Recall İçin Hiçbir Şey Feda Edilmez
v3.0'ın recall'ı (0.98) = altın değerinde

### 4. Ensemble Magic Yoktur
10 yöntem denendi, hiçbiri işe yaramadı

### 5. Simple is Beautiful
24 feature + default LightGBM > 68 feature + stacking

---

## 🚫 "Neden X'i Denemediler?"

✅ **Feature Engineering** - Denendi, başarısız

✅ **Farklı Algoritmalar** - 5 algoritma test edildi

✅ **Ensemble** - 10 yöntem denendi

✅ **Hyperparameter Tuning** - Optuna ile yapıldı

❌ **Daha Fazla Veri** - İmkan yok (en etkili olurdu)

❌ **Deep Learning** - Veri yetersiz, overkill

❌ **GNN** - Infrastructure yok

---

## ⚠️ Critical Findings

### ExtraTrees Paradoksu
- Validation AUC: 0.8106 (orta)
- **Test AUC: 0.7751 (en iyi!)**
- Grid search onu hiç seçmedi!
- Validation-test mismatch

### Recall Sorunu
Hiçbir optimizasyon v3.0'ın recall'ını yakalayamadı:
- v3.0: 0.98 (100'de 2 kayıp)
- En iyi diğer: 0.85 (100'de 15 kayıp)
- **13 müşteri farkı = ciddi gelir kaybı**

---

## 💡 İş Değeri Perspektifi

### v3.0 Kullanımı

**Olasılık Skorları:**
```
Müşteri A: 0.95 → Kesin kampanya gönder
Müşteri B: 0.75 → Orta öncelik
Müşteri C: 0.55 → İndirim göster
Müşteri D: 0.25 → Hiç uğraşma
```

**Avantaj:**
- Müşterileri sıralayabilme
- Budget optimizasyonu
- Dinamik strateji

---

## ✅ Final Karar

### v3.0 Baseline Kullanılacak

**Neden:**
1. En yüksek F1 (0.69)
2. En yüksek Recall (0.98) ⭐
3. En yüksek Precision (0.65)
4. En düşük Gap (11%)
5. Basit ve maintainable

**İş Değeri:**
- 100 müşteriden 98'ini yakalıyor
- Minimal false negative
- ROI maksimum

---

## 🔮 Gelecek Öneriler

### Eğer Kaynak Bulunursa:

**1. Daha Fazla Veri (+2-3% AUC)**
- 10M+ session hedef
- En etkili iyileştirme

**2. External Features (+1-2% AUC)**
- Ürün kategorisi detayları
- Fiyat trendleri
- Mevsimsellik

**3. A/B Testing**
- Gerçek kullanıcılarla test
- Business metric (ROI) track

---

## 📈 Proje Zaman Çizelgesi

- **Hafta 1-2:** v3.0 analiz, feature engineering
- **Hafta 3:** Algorithm testing
- **Hafta 4-5:** Hyperparameter optimization (Optuna)
- **Hafta 6-7:** 10 ensemble yöntemi
- **Hafta 8:** Final analiz ve karar

**Toplam:** ~20 saat computation

---

## 🎯 Sonuç

**10 farklı optimizasyon denendi.**

**Hiçbiri v3.0'dan iyi değil.**

**Bu başarısızlık DEĞİL, sistematik optimizasyon!**

Her deneme bize bir şey öğretti:
- Veri kalitesi #1 faktör
- Basit modeller güçlü olabilir
- Tek metrik optimize etmek riskli
- Validation-test gap kritik

---

## 📊 Teknik Detaylar

**Veri:**
- Train: 2.2M sessions
- Val: 469K sessions
- Test: 541K sessions
- Features: 24

**Araçlar:**
- Python 3.14
- LightGBM, XGBoost, Scikit-learn
- Optuna
- Google Colab (parallel execution)

---

## 📝 Kaynaklar

**Raporlar:**
- `FINAL_PROJECT_REPORT.md` - Kapsamlı rapor
- `reports/final_report_v3.md` - v3.0 detayları
- `reports/phase3_detailed_metrics.csv` - Tüm metrikler
- `reports/phase4c_multiobjective_results.csv` - Ensemble sonuçları

**Modeller:**
- `models/` klasöründe tüm modeller
- v3.0 GitHub'da mevcut

---

## 🙏 Teşekkürler

**Sorular?**

---

## 📎 Ek: Metrik Açıklamaları

**AUC (Area Under ROC Curve):**
- Model'in sıralama yeteneği
- 0.5 = Rastgele, 1.0 = Mükemmel
- v3.0: 0.7619 = İyi

**F1 Score:**
- Precision ve Recall'ın harmonik ortalaması
- Dengeli performans göstergesi
- v3.0: 0.69 = Çok iyi

**Recall:**
- Tüm pozitif örneklerin yakalanma oranı
- v3.0: 0.98 = Neredeyse mükemmel!
- İş değeri açısından en kritik

**Precision:**
- Pozitif tahminlerin doğruluğu
- v3.0: 0.65 = İyi

**Gap:**
- Train-Test AUC farkı
- Overfitting göstergesi
- v3.0: 11% = Düşük (iyi!)
