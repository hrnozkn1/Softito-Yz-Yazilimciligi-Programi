# NLP Projeleri — Softito Yapay Zeka Programı

NLP temellerinden Transformer mimarisine uzanan 6 aşamalı bir seri. Her proje tek bir `.py` dosyasıdır: çalıştır, görselleri oluştur, sonuçları karşılaştır.

## Seri İçeriği

| # | Proje | Veri | Model |
|---|-------|------|-------|
| 01 | [`01-tf-idf/`](01-tf-idf/) | TTC-4900 (Türkçe, 7 sınıf) | TF-IDF + LogisticRegression |
| 02 | [`02-word-embeddings/`](02-word-embeddings/) | TTC-4900 (Türkçe, 7 sınıf) | Word2Vec, FastText, TF-IDF karşılaştırması + t-SNE |
| 03 | [`03-rnn/`](03-rnn/) | TTC-4900 (Türkçe, 7 sınıf) | Vanilla RNN (PyTorch `nn.RNN`) |
| 04 | [`04-lstm/`](04-lstm/) | AG News (İngilizce, 4 sınıf, 120K) | LSTM (PyTorch `nn.LSTM`) |
| 05 | [`05-attention/`](05-attention/) | AG News (İngilizce, 4 sınıf, 120K) | BiLSTM + Additive Attention |
| 06 | [`06-transformer/`](06-transformer/) | AG News (İngilizce, 4 sınıf, 120K) | Transformer Encoder (`nn.TransformerEncoder`) |

## Mimari Gelişimi

```
TF-IDF (istatistiksel)
   ↓
Word2Vec / FastText (gömmeler)
   ↓
Vanilla RNN (tekrarlayan, vanishing gradient)
   ↓
LSTM (vanishing gradient çözümü, uzun dönem hafıza)
   ↓
LSTM + Attention (tüm çıktılara odaklanma)
   ↓
Transformer (self-attention, paralel işleme, sıra dışı)
```

## Veri Setleri

| Veri | Projeler | Dil | Sınıf | Boyut |
|------|----------|-----|-------|-------|
| [TTC-4900](https://www.kaggle.com/datasets/savasy/ttc4900) | 01, 02, 03 | 🇹🇷 Türkçe | 7 | 4.900 |
| [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) | 04, 05, 06 | 🇬🇧 İngilizce | 4 | 120.000 |

İlk 3 proje Türkçe haber verisiyle (TTC-4900) yapıldı; son 3 proje uluslararası benchmark (AG News) ile devam etti.

## Nasıl Çalıştırılır

```bash
# Sanal ortam (Python 3.12+)
source /tmp/venv312/bin/activate

# Projelerden birine gir, çalıştır
cd nlp/04-lstm
pip install -r requirements.txt
python lstm_siniflandirma.py
```

Her script:
1. Veriyi `data/` klasöründe arar; bulamazsa indirme talimatı gösterir + sentetik fallback sunar
2. EDA (sınıf dağılımı, token histogramı, sık kelimeler) → `figures/01_eda.png`
3. Model eğitimi ve değerlendirme
4. Training curves → `figures/02_training_curves.png`
5. Confusion matrix → `figures/03_confusion_matrix.png`

## Klasör Yapısı (Her Proje İçin)

```
nlp/XX-proje/
├── README.md              # Proje dokümantasyonu
├── requirements.txt       # Bağımlılıklar
├── .gitignore             # data/ hariç
├── data/README.md         # Veri indirme talimatı
├── figures/               # Görsel çıktılar
└── proje_adi.py           # Ana script
```

## Gereksinimler

- Python 3.12+ (gensim için Python <3.14 gerekir)
- torch, scikit-learn, pandas, matplotlib, seaborn, tqdm, numpy, datasets
