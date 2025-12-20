# Data Directory

Bu klasörde işlenmiş session-level veriler bulunur.

## Dosyalar

- `train_sessions.parquet` - Eğitim verisi (session-level)
- `val_sessions.parquet` - Validasyon verisi (session-level)
- `test_sessions.parquet` - Test verisi (session-level)

## Veri Boyutu

Büyük dosyalar olduğu için GitHub'a yüklenmemiştir. 

## Veriyi Oluşturma

Veriyi oluşturmak için:

```bash
python -m src.data.prepare
python -m src.features.build
```

