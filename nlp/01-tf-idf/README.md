# TF-IDF: Teoriden Pratiğe

TF-IDF (Term Frequency — Inverse Document Frequency) konusunu, en temel matematiksel formüllerden gerçek veri ile sınıflandırmaya ve boyut indirgeme analizine kadar kapsamlı bir şekilde ele alan tek dosyalık bir proje.

## İçindekiler

| Bölüm | Başlık | Açıklama |
|-------|--------|----------|
| 1 | **Teori** | TF, IDF, TF-IDF formülleri ve yorumları |
| 2 | **Manuel Hesaplama** | 4 Türkçe dokümanla adım adım TF-IDF tablosu, "kedi" özel hesabı, nadir/sık kelime karşılaştırması |
| 3 | **Scikit-learn ile TF-IDF** | `TfidfVectorizer` kullanımı, ısı haritası görselleştirmesi |
| 4 | **TTC-4900 Türkçe Veri Seti** | Gerçek Türkçe haber verisi, cosine similarity, sınıf dağılımı |
| 5 | **Sınıflandırma** | TF-IDF + Logistic Regression ile 7 kategorili Türkçe haber sınıflandırması |
| 6 | **SVD Analizi** | Boyut indirgeme (10.000 → 25-500) ve performans takası analizi |
| 7 | **TF-IDF Limitleri** | Anlamsal ilişki, bağlam kaybı, seyreklik, boyut laneti — görsellerle |
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

**TTC-4900** (HuggingFace: `savasy/ttc4900`):
- 7 kategori: siyaset, dünya, ekonomi, kültür, sağlık, spor, teknoloji
- Her kategoride 700 haber, toplam 4.900 örnek
- Veri otomatik indirilir, internet yoksa sentetik fallback ile çalışır

## Sonuçlar

| Model | Boyut | Doğruluk | F1 (macro) |
|-------|-------|----------|------------|
| TF-IDF + Logistic Regression | 10.000 | **0.98+** | **0.98+** |
| TF-IDF + SVD(500) + LogReg | 500 | ~0.97 | ~0.97 |
| TF-IDF + SVD(200) + LogReg | 200 | ~0.97 | ~0.97 |
| TF-IDF + SVD(100) + LogReg | 100 | ~0.97 | ~0.97 |
| TF-IDF + SVD(50) + LogReg | 50 | ~0.96 | ~0.96 |
| TF-IDF + SVD(25) + LogReg | 25 | ~0.95 | ~0.95 |

**Çıkarım:** SVD ile öznitelik uzayı %99.75 sıkıştırıldığında (10.000 → 25) doğruluk yalnızca ~3 puan düşüyor.

## Görseller

| Görsel | Açıklama |
|--------|----------|
| `figures/01_tfidf_heatmap.png` | TF-IDF matrisi ısı haritası |
| `figures/02_veri_dagilimi.png` | Sınıf dağılımı bar chart |
| `figures/03_token_uzunluklari.png` | Haber uzunluklarının histogramı |
| `figures/04_confusion_matrix.png` | Sınıflandırma confusion matrix |
| `figures/05_kelime_onemi.png` | Her kategori için en karakteristik kelimeler |
| `figures/06_svd_tradeoff.png` | SVD performans takası grafiği |
| `figures/07_sparsity.png` | Vocabulary büyüklüğüne karşı seyreklik |
| `figures/08_boyut_laneti.png` | Boyut laneti gösterimi |

## Proje Yapısı

```
nlp/01-tf-idf/
├── tfidf_kapsamli.py        # Ana Python scripti
├── README.md                # Bu dosya
├── requirements.txt         # Bağımlılıklar
└── figures/                 # Otomatik oluşturulan görseller
```
