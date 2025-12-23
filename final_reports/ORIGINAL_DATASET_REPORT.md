# Orijinal Veri Seti Raporu
## E-Commerce Purchase Prediction - Ham Veri Analizi

**Rapor Tarihi:** 23 Aralık 2025  
**Veri Kaynağı:** `archive/` klasörü  
**Format:** Apache Parquet

---

## 📋 Genel Bakış

### Veri Seti Boyutu

| Özellik | Değer |
|---------|-------|
| **Toplam Satır** | 11,495,242 event (11.5M+) |
| **Toplam Kolon** | 19 |
| **Dosya Boyutu** | ~599 MB (train.parquet) |
| **Veri Seviyesi** | Event-level (Her satır bir kullanıcı etkileşimi) |
| **Kayıt Dönemi** | 2020 Ocak - Mayıs |

### Dosya Yapısı

```
archive/
├── train.parquet       (599 MB - Eğitim verisi)
├── val.parquet         (133 MB - Validasyon verisi)
└── test.parquet        (152 MB - Test verisi)
```

---

## 📊 Veri Kolonları Detaylı Analizi

### **1. Zaman Bilgileri (Temporal Features)**

#### `event_time` 
- **Tip:** Object (String timestamp)
- **Açıklama:** Kullanıcı etkileşiminin gerçekleştiği tam zaman
- **Format:** ISO 8601 format (örn: "2020-01-01 12:30:45 UTC")
- **Kullanım:** Session oluşturma, temporal pattern analizi
- **Örnek:** `"2020-01-07 09:55:24 UTC"`

#### `timestamp`
- **Tip:** datetime64[us, UTC]
- **Açıklama:** event_time'ın datetime formatı
- **Kullanım:** Daha hızlı datetime işlemleri için optimize edilmiş
- **Avantaj:** Pandas datetime fonksiyonları ile uyumlu

#### `ts_hour` (0-23)
- **Tip:** int16
- **Açıklama:** Etkileşimin gerçekleştiği saat
- **Kullanım:** Gün içi aktivite patternleri
- **İş Değeri:** Peak saatleri belirlemek (örn: 18-21 arası yoğunluk)

#### `ts_minute` (0-59)
- **Tip:** int16
- **Açıklama:** Etkileşimin gerçekleştiği dakika
- **Kullanım:** Daha detaylı temporal analiz

#### `ts_weekday` (0-6)
- **Tip:** int16
- **Açıklama:** Haftanın günü (0=Pazartesi, 6=Pazar)
- **İş Değeri:** Hafta içi/sonu davranış farkları

#### `ts_day` (1-31)
- **Tip:** int16
- **Açıklama:** Ayın günü
- **Kullanım:** Aylık pattern analizi (maaş günü etkisi vs.)

#### `ts_month` (1-12)
- **Tip:** int16
- **Açıklama:** Ay numarası
- **Kullanım:** Mevsimsel trend analizi

#### `ts_year`
- **Tip:** int16
- **Açıklama:** Yıl (2020)
- **Not:** Tek yıllık veri olduğu için varyasyon yok

---

### **2. Kullanıcı Bilgileri (User Features)**

#### `user_id`
- **Tip:** Object (String)
- **Açıklama:** Kullanıcının benzersiz kimliği
- **Unique Değerler:** ~3.2M farklı kullanıcı
- **Kullanım:** Kullanıcı bazlı aggregation, session oluşturma
- **Örnek:** `"100037567"`

#### `user_session`
- **Tip:** Object (String)
- **Açıklama:** Kullanıcının o anki session ID'si
- **Format:** `{user_id}_{session_number}`
- **Unique Değerler:** ~3.7M farklı session
- **Kullanım:** Event'leri session'lara gruplama
- **Örnek:** `"100037567_1"` (user 100037567'nin 1. session'ı)
- **Önemli:** Her session genelde 20-30 dk içindeki işlemleri içerir

---

### **3. Ürün Bilgileri (Product Features)**

#### `product_id`
- **Tip:** Object (String)
- **Açıklama:** Ürünün benzersiz kimliği
- **Unique Değerler:** ~235,000 farklı ürün
- **Kullanım:** Ürün bazlı analiz, diversity hesaplama
- **Format:** Sayısal string ID

#### `brand`
- **Tip:** Object (String)
- **Açıklama:** Ürünün markası
- **Unique Değerler:** ~5,200 farklı marka
- **Kullanım:** Marka tercihi analizi
- **Özellik:** Bazı ürünlerde eksik olabilir (null)

#### `price`
- **Tip:** Object (String - sayısal değer)
- **Açıklama:** Ürün fiyatı (muhtemelen USD/EUR)
- **Range:** 0.01 - 50,000+ (çok geniş aralık)
- **Kullanım:** Fiyat davranışı analizi, ortalama sepet değeri
- **Not:** String olarak saklanmış, float'a dönüştürülmeli

---

### **4. Kategori Bilgileri (Category Hierarchy)**

E-commerce ürün kategorileri hiyerarşik yapıda (tree structure):

#### `cat_0` (Ana Kategori)
- **Tip:** Object (String)
- **Açıklama:** En üst seviye kategori
- **Unique Değerler:** ~17 ana kategori
- **Örnekler:** "electronics", "appliances", "apparel" vb.

#### `cat_1` (Alt Kategori 1)
- **Tip:** Object (String)
- **Açıklama:** İkinci seviye kategori
- **Unique Değerler:** ~90 alt kategori
- **Örnekler:** "smartphone", "audio", "computers.notebook"

#### `cat_2` (Alt Kategori 2)
- **Tip:** Object (String)
- **Açıklama:** Üçüncü seviye kategori
- **Unique Değerler:** ~300+ kategori
- **Detay:** Daha spesifik ürün grupları

#### `cat_3` (Alt Kategori 3)
- **Tip:** Object (String)
- **Açıklama:** En detaylı kategori seviyesi
- **Unique Değerler:** ~600+ kategori
- **Not:** En spesifik ürün tipi

**Kategori Hiyerarşisi Örneği:**
```
cat_0: electronics
  └── cat_1: smartphone
      └── cat_2: smartphone.android
          └── cat_3: smartphone.android.flagship
```

---

### **5. Etkileşim Bilgileri (Interaction Features)**

#### `event_type`
- **Tip:** Object (String/Categorical)
- **Açıklama:** Kullanıcının yaptığı etkileşim tipi
- **Değerler:**
  - `"view"` - Ürünü görüntüleme (~95% çoğunlukta)
  - `"cart"` - Sepete ekleme (~3-4%)
  - `"purchase"` - Satın alma (~1-2%)
  - `"remove_from_cart"` - Sepetten çıkarma (nadir)
  
**Funnel Yapısı:**
```
View (1000 kişi)
  → Cart (30-40 kişi)
    → Purchase (10-15 kişi)
```

**Dönüşüm Oranları:**
- View → Cart: ~3-4%
- Cart → Purchase: ~30-40%
- View → Purchase: ~1-1.5% (direkt dönüşüm)

---

### **6. Hedef Değişken (Target Variable)**

#### `target`
- **Tip:** int64 (Binary: 0 veya 1)
- **Açıklama:** O session'da alışveriş yapıldı mı?
- **Değerler:**
  - `0` - Alışveriş yapılmadı (~85-90%)
  - `1` - Alışveriş yapıldı (~10-15%)
  
**Sınıf Dağılımı:**
- **Pozitif (Purchase):** ~10-15%
- **Negatif (No Purchase):** ~85-90%
- **Imbalance Oranı:** ~6:1 (negatif:pozitif)

**Önemli Not:** Bu imbalance, modelin precision/recall dengesinde kritik rol oynar.

---

## 🗄️ Neden Apache Parquet?

### **1. Sıkıştırma Verimliliği**

| Format | Boyut | Sıkıştırma Oranı |
|--------|-------|------------------|
| CSV | ~2.5 GB | 1x (baseline) |
| **Parquet** | **~599 MB** | **~4x daha küçük** |

**Kazanç:** 
- Disk alanı tasarrufu: 1.9 GB
- Daha hızlı veri transferi
- Daha az I/O işlemi

---

### **2. Hızlı Okuma Performansı**

**Columnar Storage Avantajı:**
```
CSV (Row-based):          Parquet (Column-based):
[user_id, price, ...]     [user_id, user_id, ...]
[user_id, price, ...]     [price, price, price...]
[user_id, price, ...]     
```

**Sonuç:**
- Sadece gerekli kolonları okuma (projection)
- Predicate pushdown (filtreleme data okumadan önce)
- **10-100x daha hızlı** sorgu performansı

**Örnek:**
```python
# Sadece price ve target kolonlarını okuma
df = pd.read_parquet('train.parquet', columns=['price', 'target'])
# CSV ile tüm dosya okunmak zorunda, Parquet ile sadece 2 kolon!
```

---

### **3. Veri Tipi Koruması**

| Feature | CSV | Parquet |
|---------|-----|---------|
| **timestamp** | String → Manual parse | datetime64 (native) |
| **ts_hour** | String/int → ambiguous | int16 (optimized) |
| **price** | String → float parsing risk | Numeric (safe) |

**Avantaj:**
- Type safety (hata riski azalır)
- Bellek optimizasyonu (int16 vs int64)
- Automatic type inference

---

### **4. Schema Evolution**

Parquet dosyaları schema bilgisi taşır:
```
Parquet Metadata:
- Column: user_id, Type: string, Nullable: false
- Column: price, Type: double, Nullable: true
- Column: timestamp, Type: timestamp[us, tz=UTC]
...
```

**Avantaj:**
- Self-documenting (kendi kendine dokümante)
- Version control (schema değişikliklerini izleme)
- Data validation (otomatik tip kontrolü)

---

### **5. Ecosystem Uyumluluğu**

Parquet tüm big data araçlarıyla uyumlu:
- ✅ Pandas
- ✅ Spark
- ✅ Dask
- ✅ PyArrow
- ✅ Presto/Athena
- ✅ BigQuery

**CSV:** Sadece temel okuma/yazma
**Parquet:** Advanced features (compression, encoding, statistics)

---

## 📈 Veri Karakteristikleri

### **Event Dağılımı**

```
view:             ~10.9M events (95%)
cart:               ~400K events (3.5%)
purchase:           ~150K events (1.3%)
remove_from_cart:    ~45K events (0.4%)
```

### **Session Özellikleri**

| Metrik | Ortalama | Medyan |
|--------|----------|--------|
| **Events/Session** | 3.1 | 2 |
| **Session Duration** | 12 dakika | 5 dakika |
| **Unique Products/Session** | 2.4 | 2 |

### **Temporal Patterns**

**En Yoğun Saatler:**
- 18:00-21:00 (akşam saatleri)
- 12:00-14:00 (öğle arası)

**En Aktif Günler:**
- Hafta sonu (Cumartesi-Pazar)
- Cuma akşamı

---

## ⚠️ Veri Kalite Notları

### **Eksik Değerler (Missing Values)**

| Kolon | Missing Rate | Açıklama |
|-------|--------------|----------|
| `brand` | ~15-20% | Bazı ürünlerde marka bilgisi yok |
| `cat_3` | ~10% | En detaylı kategori bazı ürünlerde eksik |
| `cat_2` | ~5% | İkinci seviye kategori nadir eksik |
| Diğer kolonlar | <1% | Çok az eksik değer |

### **Veri Tutarlılığı**

✅ **İyi Yönler:**
- User ID'ler tutarlı
- Timestamp'ler sıralı
- Event types standardize

⚠️ **Dikkat Edilmesi Gerekenler:**
- Price outlier'ları var (0.01'den 50,000'e kadar)
- Bazı session'lar çok kısa (<10 saniye)
- Bazı session'lar çok uzun (>2 saat - muhtemelen bot)

---

## 🔄 Veri İşleme İhtiyaçları

Bu ham veri, makine öğrenmesi için **event-level**'dan **session-level**'a dönüştürülmelidir:

### **Gerekli Transformasyonlar:**

1. **Session Aggregation**
   - Event'leri session'lara gruplama
   - Her session için özet metrikler oluşturma

2. **Feature Engineering**
   - Temporal features (session süresi, event rate)
   - Price statistics (mean, min, max, std)
   - Category diversity
   - Event type distribution

3. **Data Cleaning**
   - Outlier'ları temizleme
   - Bot detection
   - Missing value handling

4. **Target Definition**
   - Session level target oluşturma
   - Çok kısa/uzun session'ları filtreleme

---

## 📝 Özet

### **Güçlü Yönler:**
- ✅ Büyük ve gerçek veri (11.5M+ events)
- ✅ Zengin feature set (19 kolon)
- ✅ Temporal coverage (5 ay)
- ✅ Parquet formatı (hızlı ve verimli)
- ✅ Category hiyerarişisi (multi-level)

### **Zorluklar:**
- ⚠️ Class imbalance (~10% pozitif)
- ⚠️ Missing values (özellikle brand)
- ⚠️ Price outlier'ları
- ⚠️ Event-level data (session transformation gerekli)

### **Veri Boyutu Karşılaştırması:**

```
Raw Events:    11.5M rows (event-level)
                    ↓ (Session aggregation)
Final Sessions: ~3.7M rows (session-level)
                    ↓ (Train/val/test split)
Training Set:   ~2.2M sessions
```

---

**Sonraki Adım:** Bu ham veri, **session-level aggregation** ile işlenmiş veri setine dönüştürülmüştür. Detaylar için `PROCESSED_DATASET_REPORT.md` dosyasına bakınız.
