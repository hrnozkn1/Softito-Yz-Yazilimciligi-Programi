# Softito YZ Yazılımcılığı Programı — NLP

NLP'ye sıfırdan başlayıp Transformer'a kadar uzanan 6 aşamalı bir seri. Her proje bağımsız çalışan tek bir Python scripti: çalıştır, eğit, grafikleri gör, karşılaştır.

## Projeler

| # | Proje | Ne Öğretir? | Veri Seti | Model |
|---|-------|-------------|-----------|-------|
| 01 | [TF-IDF](nlp/01-tf-idf/) | Kelimeleri sayıya dökme, vektör uzayı | [TTC-4900](https://www.kaggle.com/datasets/savasy/ttc4900) (Türkçe haber, 7 kategori) | TF-IDF + LogisticRegression |
| 02 | [Word Embeddings](nlp/02-word-embeddings/) | Anlamsal gömmeler, kelime benzerliği | TTC-4900 | Word2Vec, FastText, TF-IDF + t-SNE |
| 03 | [Vanilla RNN](nlp/03-rnn/) | Tekrarlayan sinir ağları, sequence modelleme | TTC-4900 | `nn.RNN` (PyTorch) — %41 accuracy |
| 04 | [LSTM](nlp/04-lstm/) | Vanishing gradient sorununun çözümü | [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) (İngilizce, 4 kategori, 120K haber) | `nn.LSTM` |
| 05 | [LSTM + Attention](nlp/05-attention/) | Modelin hangi kelimelere odaklandığını görmek | AG News | BiLSTM + Bahdanau Attention |
| 06 | [Transformer](nlp/06-transformer/) | Self-attention, paralel işleme, sıra bağımsızlık | AG News | `nn.TransformerEncoder` |

### Mimari Gelişimi

```
TF-IDF ──→ Word2Vec ──→ RNN ──→ LSTM ──→ LSTM+Attn ──→ Transformer
(1990s)    (2013)      (1986)   (1997)    (2015)        (2017)
```

### Kullanılan Veri Setleri

| Veri Seti | Dil | Sınıflar | Örnek Sayısı | Kullanıldığı Projeler |
|-----------|-----|----------|-------------|----------------------|
| [TTC-4900](https://www.kaggle.com/datasets/savasy/ttc4900) | 🇹🇷 Türkçe | siyaset, ekonomi, kültür, sağlık, spor, teknoloji, dünya | 4.900 | 01, 02, 03 |
| [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) | 🇬🇧 İngilizce | World, Sports, Business, Sci/Tech | 120.000 | 04, 05, 06 |

TTC-4900 dengeli bir Türkçe haber verisi (her sınıftan 700 haber). AG News ise uluslararası bir benchmark — bu sayede ilk 3 projede Türkçe NLP deneyimi kazanılır, sonraki 3 projede daha büyük bir veriyle mimariler karşılaştırılır.

## Nasıl Çalıştırılır

```bash
git clone https://github.com/hrnozkn1/Softito-Yz-Yazilimciligi-Programi.git
cd Softito-Yz-Yazilimciligi-Programi/nlp/04-lstm
pip install -r requirements.txt
python lstm_siniflandirma.py
```

Her script çalıştırıldığında:
1. Veriyi `data/` klasöründe arar, yoksa indirme adresini söyler (sentetik fallback verisiyle de çalışır)
2. EDA grafiği basar: sınıf dağılımı, token uzunluğu histogramı, en sık kelimeler
3. Modeli eğitir ve her epoch sonu accuracy + F1 gösterir
4. Eğitim eğrilerini ve confusion matrix'i `figures/` klasörüne kaydeder

### Klasör Yapısı

```
nlp/XX-proje/
├── README.md
├── requirements.txt
├── .gitignore              # data/ hariç (gitignored)
├── data/README.md          # veri seti bilgisi
├── figures/                # çıktı grafikleri
└── proje_adi.py            # tek dosya, çalıştır ve gör
```

## Gereksinimler

- Python 3.12+ önerilir (gensim 3.14'te derlenmiyor)
- PyTorch, scikit-learn, pandas, matplotlib, seaborn, tqdm, datasets, numpy

## Notlar

- 03-rnn (Vanilla RNN) test edildi ve sonuçları push edildi — %41 accuracy
- 04-05-06 scriptleri hazır ancak eğitim tam sonuç için ~15-20 dk sürer (CPU'da çalıştırılmalı)
- GPU varsa `DEVICE` otomatik algılanır, eğitim çok daha hızlı olur
