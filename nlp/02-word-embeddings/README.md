# Word Embeddings: Word2Vec, FastText ve TF-IDF ile Karşılaştırma

Word Embeddings, kelimeleri anlamca yakın olanların birbirine yakın durduğu yoğun vektörlerle temsil eden bir tekniktir. TF-IDF'in "anlam" ve "bağlam" eksikliğini giderir. Bu projede Word2Vec, FastText ve TF-IDF karşılaştırmalı olarak incelenir.

## İçindekiler

| Bölüm | Başlık | Açıklama |
|-------|--------|----------|
| 1 | **Teori** | One-hot → Embedding mantığı, Distributional Hypothesis |
| 2 | **Veri Yükleme** | HuggingFace Keloğlan Türkçe Sentiment (630K+ yorum, 3 sınıf) |
| 3 | **Word2Vec** | Skip-gram eğitimi, benzer kelime örnekleri |
| 4 | **FastText** | Subword bilgisi, OOV (Out-of-Vocabulary) testi |
| 5 | **t-SNE Görselleştirme** | Kelime vektörlerinin 2B uzayda görselleştirilmesi + Analoji testleri |
| 6 | **Sınıflandırma** | TF-IDF vs Word2Vec vs FastText (Logistic Regression ile) |
| 7 | **Özet** | Yöntem karşılaştırması ve ne zaman hangisi kullanılır |

## Çalıştırma

```bash
pip install -r requirements.txt
python word_embeddings_kapsamli.py
```

Script çalıştığında:
- Tüm adımlar konsola yazdırılır
- Görseller `figures/` klasörüne kaydedilir (PNG)
- İlk çalıştırmada HuggingFace veri seti indirilir (~630K örnek)

## Veri Seti

**Keloğlan Turkish Sentiment Analysis Dataset** (`engin1123/keloglan-turkish-sentiment-analysis-dataset`):
- 599.607 eğitim + 31.559 validasyon örneği
- 3 sınıf: Negatif (0), Nötr (1), Pozitif (2)
- Kaynak: e-ticaret, film ve ürün yorumları
- Lisans: CC-BY-NC 4.0
- Boyut: ~630K Türkçe yorum

## Sonuçlar

| Model | Boyut | Accuracy | F1 (weighted) |
|-------|-------|----------|---------------|
| TF-IDF + Logistic Regression | 5.000 | **0.852** | **0.849** |
| Word2Vec + Logistic Regression | 100 | 0.829 | 0.826 |
| FastText + Logistic Regression | 100 | 0.836 | 0.833 |

**Çıkarım:** Embedding yöntemleri daha az boyutla (~%98 daha az) TF-IDF'e yakın performans gösterir. FastText, OOV kelimeleri tahmin edebildiği için Türkçe gibi sondan eklemeli dillerde Word2Vec'ten hafif üstündür.

## Görseller

| Görsel | Açıklama |
|--------|----------|
| `figures/02_tsne_word2vec.png` | Word2Vec kelime vektörlerinin t-SNE ile 2B görselleştirmesi |
| `figures/03_classification_comparison.png` | TF-IDF / Word2Vec / FastText sınıflandırma performans karşılaştırması |

## Proje Yapısı

```
nlp/02-word-embeddings/
├── word_embeddings_kapsamli.py   # Ana Python scripti
├── README.md                     # Bu dosya
├── requirements.txt              # Bağımlılıklar
└── figures/                      # Otomatik oluşturulan görseller
```

## Kaynaklar

- Mikolov et al. "Efficient Estimation of Word Representations in Vector Space" (2013)
- Bojanowski et al. "Enriching Word Vectors with Subword Information" (2016) — FastText
- Keloğlan Turkish Sentiment Dataset: https://huggingface.co/datasets/engin1123/keloglan-turkish-sentiment-analysis-dataset
