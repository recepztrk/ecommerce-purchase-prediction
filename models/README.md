# Models Directory

Bu klasörde eğitilmiş modeller ve sonuçlar bulunur.

## Model Dosyaları (GitHub'da yok - çok büyük)

- `lightgbm_model.txt` - LightGBM modeli
- `logistic_regression.pkl` - Logistic Regression modeli
- `scaler.pkl` - StandardScaler
- `training_results.pkl` - Tüm eğitim sonuçları

## Sonuç Dosyaları (GitHub'da var)

- `feature_importance.csv` - Özellik önem sıralaması
- `model_comparison.csv` - Model karşılaştırma tablosu

## Modelleri Eğitme

Modelleri yeniden eğitmek için:

```bash
python -m src.models.train
python -m src.evaluation.evaluate
```

## Model Performansı

- **En İyi Model:** LightGBM
- **Test ROC-AUC:** 0.5936
- **Test F1:** 0.6272 (threshold=0.40)
- **Recall:** 0.9960

