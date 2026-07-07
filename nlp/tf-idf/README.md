# TF-IDF: Teoriden Pratiğe

TF-IDF (Term Frequency — Inverse Document Frequency) konusunu, en temel matematiksel formüllerden gerçek veri ile sınıflandırmaya ve boyut indirgeme analizine kadar kapsamlı bir şekilde ele alan tek dosyalık bir proje.

## İçindekiler

| Bölüm | Başlık | Açıklama |
|-------|--------|----------|
| 1 | **Teori** | TF, IDF, TF-IDF formülleri ve yorumları |
| 2 | **Manuel Hesaplama** | 4 Türkçe dokümanla adım adım TF-IDF tablosu, "kedi" özel hesabı, nadir/sık kelime karşılaştırması |
| 3 | **Scikit-learn ile TF-IDF** | `TfidfVectorizer` kullanımı, ısı haritası görselleştirmesi |
| 4 | **20 Newsgroups** | Gerçek veri seti ile TF-IDF vektörleştirme ve cosine similarity |
| 5 | **Sınıflandırma** | TF-IDF + Logistic Regression ile 4 kategorili haber sınıflandırması (%84+ doğruluk) |
| 6 | **SVD Analizi** | Boyut indirgeme (10.000 → 25-500) ve performans takası analizi |
| 7 | **TF-IDF Limitleri** | Semantik ilişki, bağlam kaybı, seyreklik, boyut laneti — görsellerle |
| 8 | **Özet & Örnek Tahmin** | Sonuç tablosu, yeni metinlerle tahmin |

## Çalıştırma

```bash
pip install -r requirements.txt
python tfidf_kapsamli.py
```

Script çalıştığında:
- Tüm adımlar konsola yazdırılır
- Görseller `figures/` klasörüne kaydedilir (PNG)

## Veri Seti

**20 Newsgroups** (`sklearn.datasets.fetch_20newsgroups`):
- 4 kategori: `rec.sport.baseball`, `sci.space`, `comp.graphics`, `talk.politics.guns`
- ~3.500 belge (eğitim + test)

## Sonuçlar

| Model | Boyut | Accuracy | F1 (macro) |
|-------|-------|----------|------------|
| TF-IDF + Logistic Regression | 10.000 | **0.981** | **0.981** |
| TF-IDF + SVD(500) + LogReg | 500 | 0.977 | 0.977 |
| TF-IDF + SVD(200) + LogReg | 200 | 0.977 | 0.977 |
| TF-IDF + SVD(100) + LogReg | 100 | 0.972 | 0.972 |
| TF-IDF + SVD(50) + LogReg | 50 | 0.963 | 0.963 |
| TF-IDF + SVD(25) + LogReg | 25 | 0.953 | 0.954 |

**Çıkarım:** SVD ile öznitelik uzayı %99.75 sıkıştırıldığında (10.000 → 25) doğruluk yalnızca ~3 puan düşüyor (0.981 → 0.953).

## Görseller

| Görsel | Açıklama |
|--------|----------|
| `figures/01_tfidf_heatmap.png` | TF-IDF matrisi ısı haritası |
| `figures/02_confusion_matrix.png` | Sınıflandırma confusion matrix |
| `figures/03_top_words.png` | En etkili pozitif/negatif kelimeler |
| `figures/04_svd_tradeoff.png` | SVD performans takası grafiği |
| `figures/05_sparsity.png` | Vocabulary büyüklüğüne karşı seyreklik |
| `figures/06_curse_of_dimensionality.png` | Boyut laneti gösterimi |

## Proje Yapısı

```
nlp/tf-idf/
├── tfidf_kapsamli.py        # Ana Python scripti
├── README.md                # Bu dosya
├── requirements.txt         # Bağımlılıklar
└── figures/                 # Otomatik oluşturulan görseller
```
