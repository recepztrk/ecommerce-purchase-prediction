# Reports Directory

Bu klasör, proje raporlarını, analiz sonuçlarını ve görselleştirmeleri içerir.

---

## 📂 Klasör Yapısı

```
reports/
├── 📄 Ana Raporlar (3 MD)
├── 📝 Ek Raporlar (3 MD)
├── 📈 CSV Sonuçlar (7 dosya)
└── 📁 final_visuals/ (10 profesyonel görsel) ⭐
```

**Toplam:** ~2.7 MB, clean & organized

---

## 📄 ANA RAPORLAR (Sunum/Rapor için ZORUNLU) ⭐⭐⭐

### 1. **FINAL_PROJECT_REPORT.md** (15 KB)
- **İçerik:** Kapsamlı proje raporu
- **Kapsam:** Tüm 10 deneme, başarısızlıklar, teknik açıklamalar
- **Kullanım:** Rapor/tez yazımı için ANA KAYNAK

### 2. **PROJECT_PRESENTATION.md** (6.5 KB)
- **İçerik:** Slide formatında sunum özeti
- **Kullanım:** 10-15 dakikalık sunum için

### 3. **VISUAL_GUIDE.md** (7.9 KB)
- **İçerik:** Görsel kullanım rehberi
- **Kullanım:** Hangi görseli nerede kullanacağına dair kılavuz

---

## 📝 EK RAPORLAR

### 4. **final_report_v3.md** (17 KB)
- **İçerik:** v3.0 baseline detaylı dokümantasyon
- **Kullanım:** v3.0 teknik detayları için referans

### 5. **final_report_v6.md** (19 KB)
- **İçerik:** v6.0 stacking ensemble raporu (reddedildi)
- **Kullanım:** Başarısızlık analizi

### 6. **failed_experiments_report.md** (13 KB)
- **İçerik:** Başarısız denemeler özeti
- **Kullanım:** Tüm başarısız denemelerin kısa özeti

---

## 📈 CSV SONUÇLAR (Tablo Verileri)

### Phase Results (7 dosya, ~6 KB)
- `phase1_29features_importance.csv` - Feature önem skorları
- `phase2_algorithm_comparison.csv` - 5 algoritma karşılaştırması
- `phase3_detailed_metrics.csv` - Detaylı metrik tablosu
- `phase3_optuna_results.csv` - Optuna optimization sonuçları
- `phase4b_alternative_ensemble_results.csv` - Ensemble yöntemleri
- `phase4c_multiobjective_results.csv` - Multi-objective sonuçlar
- `feature_importance_v4.csv` - v4.0 feature importance

**Kullanım:** Excel'de açılabilir, raporlarda tablo oluşturmak için

---

## 📊 GÖRSELLER - final_visuals/ ⭐⭐⭐

**Konum:** `reports/final_visuals/`

### **10 Profesyonel Görsel (2.5 MB)**

**Temel Set (1-6):**
1. Model Comparison Table
2. Confusion Matrix (v3.0)
3. ROC Curve
4. Feature Importance (Top 15)
5. AUC Comparison Bar Chart
6. Business Impact (98/100 capture)

**Ek Set (7-10):**
7. Data Transformation Flow (11.5M→2.2M)
8. Failed Experiments Timeline
9. Precision-Recall Curve
10. Class Distribution (Imbalance)

**Detaylar:** `final_visuals/README.md`

**Özellikler:**
- Python matplotlib/seaborn ile oluşturuldu
- 300 DPI yüksek kalite
- Professional & clean stil
- Sunum ve rapor için hazır

---

## 🎯 Kullanım Rehberi

### 📋 Rapor Yazarken

**Executive Summary:**
- FINAL_PROJECT_REPORT.md - Özet
- final_visuals/06_business_impact.png

**Problem Definition:**
- final_visuals/10_class_distribution.png

**Methodology:**
- final_visuals/07_data_transformation_flow.png
- final_visuals/04_feature_importance.png

**Results:**
- FINAL_PROJECT_REPORT.md - Detaylı sonuçlar
- final_visuals/01_model_comparison_table.png
- final_visuals/02_confusion_matrix.png
- final_visuals/03_roc_curve.png
- final_visuals/05_auc_comparison.png
- Tüm CSV dosyalar → Tablo verileri

**Discussion:**
- final_visuals/08_failed_experiments_timeline.png
- failed_experiments_report.md

**Conclusion:**
- FINAL_PROJECT_REPORT.md - Key Learnings

---

### 📽️ Sunum Hazırlarken

**Slide Yapısı:** (PROJECT_PRESENTATION.md'yi takip et)

**Slide 1-2:** Başlık + Proje Özeti

**Slide 3:** Veri Pipeline
- final_visuals/07_data_transformation_flow.png

**Slide 4:** Özellikler
- final_visuals/04_feature_importance.png

**Slide 5-6:** v3.0 Performansı
- final_visuals/02_confusion_matrix.png
- final_visuals/03_roc_curve.png

**Slide 7-8:** Model Karşılaştırması
- final_visuals/01_model_comparison_table.png
- final_visuals/05_auc_comparison.png

**Slide 9:** Deneme Süreci
- final_visuals/08_failed_experiments_timeline.png

**Slide 10:** Sonuç ve İş Etkisi
- final_visuals/06_business_impact.png

---

## 📊 Dosya İstatistikleri

### Tipler

| Tip | Sayı | Toplam Boyut |
|-----|------|--------------|
| **Markdown (.md)** | 7 | ~95 KB |
| **CSV (.csv)** | 7 | ~6 KB |
| **PNG (.png)** | 10 | ~2.5 MB (final_visuals/) |
| **README** | 2 | ~13 KB |

**Toplam:** 26 dosya, ~2.7 MB

---

### Öncelik

| Öncelik | Dosyalar | Kullanım |
|---------|----------|----------|
| **⭐⭐⭐ Kritik** | FINAL_PROJECT_REPORT.md, PROJECT_PRESENTATION.md, final_visuals/ (10 görsel) | Sunum/rapor ZORUNLU |
| **⭐⭐ Yüksek** | CSV sonuçları, final_report_v3.md | Rapor detayları |
| **⭐ Orta** | Diğer MD dosyalar | Referans |

---

## ✅ Temizlik Durumu

**Son Temizlik:** 23 Aralık 2025

**Silindi:**
- 11 eski PNG dosyası (~1.8 MB)
- feature_analysis/ klasörü
- Pickle dosyalar (daha önce)
- Log dosyalar (daha önce)

**Kazanç:** ~2 MB

**Durum:** ✅ Tamamen temiz, minimum ve organized!

---

## 🎨 Görsel Kullanım İpuçları

### Rapor İçin
- Yüksek kalite PNG'ler kullan (300 DPI)
- Her görselin caption'ını yaz
- Metin içinde referans ver (Figure 1, Figure 2)

### Sunum İçin
- Fazla görsel kalabalığı yapma (max 1 görsel/slide)
- Büyük fontlar kullan
- Görseli açıkla, okutma

---

## 🔗 İlgili Dosyalar

- **Veri Raporları:** `../final_reports/ORIGINAL_DATASET_REPORT.md`, `PROCESSED_DATASET_REPORT.md`
- **Model Dosyaları:** `../models/README.md`
- **Kaynak Kod:** `../src/README.md`
- **Ana README:** `../README.md`

---

**Son Güncelleme:** 23 Aralık 2025  
**Durum:** Clean & Organized ✅  
**Toplam Dosya:** 26  
**Toplam Boyut:** ~2.7 MB  
**Görsel Set:** Complete (10 professional visuals) ✅
