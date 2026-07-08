# NLP Projeleri

NLP'nin temel kavramlarından başlayıp günümüz mimarilerine uzanan 6 proje. Her biri bağımsız, tek dosyalık bir Python scripti.

## İçindekiler

1. [TF-IDF - Kelimeleri Sayıya Dökmek](#01-tf-idf)
2. [Word Embeddings - Anlamsal Gömmeler](#02-word-embeddings)
3. [Vanilla RNN - İlk Tekrarlayan Ağ](#03-vanilla-rnn)
4. [LSTM - Uzun Dönem Hafıza](#04-lstm)
5. [LSTM + Attention - Odaklanma Mekanizması](#05-lstm--attention)
6. [Transformer - Self-Attention ile Çağ Atlama](#06-transformer)

---

### 01 - TF-IDF

**Dosya:** [`01-tf-idf/tfidf_kapsamli.py`](01-tf-idf/tfidf_kapsamli.py)

TF-IDF'in ne olduğundan başlayıp LogisticRegression ile sınıflandırmaya, boyut indirgemeye ve kelime bulutuna kadar her şeyi tek bir scriptte anlatır. Türkçe haber verisi (TTC-4900) üzerinde çalışır.

### 02 - Word Embeddings

**Dosya:** [`02-word-embeddings/word_embeddings_karsilastirma.py`](02-word-embeddings/word_embeddings_karsilastirma.py)

Word2Vec (CBOW + Skip-gram), FastText ve TF-IDF'i aynı veri üzerinde karşılaştırır. t-SNE ile kelime vektörlerini 2 boyuta indirip görselleştirir. Hangi yöntemin ne zaman işe yaradığını gösterir.

### 03 - Vanilla RNN

**Dosya:** [`03-rnn/rnn_haber_siniflandirma.py`](03-rnn/rnn_haber_siniflandirma.py)

PyTorch ile ilk sinir ağı modeli. `nn.RNN` kullanarak Türkçe haberleri 7 kategoriden birine sınıflandırır. **%41 accuracy** — random'dan (%14) iyi, ancak Vanilla RNN'in sınırlarını (vanishing gradient) gösterir.

### 04 - LSTM

**Dosya:** [`04-lstm/lstm_siniflandirma.py`](04-lstm/lstm_siniflandirma.py)

Vanishing gradient sorununu çözen LSTM mimarisi. AG News verisiyle 4 sınıflı İngilizce haber sınıflandırması. 2 katmanlı LSTM, dropout ve gradient clipping ile daha stabil eğitim.

### 05 - LSTM + Attention

**Dosya:** [`05-attention/attention_siniflandirma.py`](05-attention/attention_siniflandirma.py)

Çift yönlü LSTM'in üstüne Bahdanau (additive) attention eklenir. Model artık son hidden state yerine tüm çıktılara bakar ve hangi tokenlara odaklandığını gösteren bir attention ağırlık grafiği basar.

### 06 - Transformer

**Dosya:** [`06-transformer/transformer_siniflandirma.py`](06-transformer/transformer_siniflandirma.py)

Self-attention mekanizması ile çalışan Transformer Encoder. RNN'lerin aksine diziyi paralel işler, positional encoding ile sıra bilgisini korur. `nn.TransformerEncoder` + adaptive pooling ile sınıflandırma.

---

## Veri Setleri

### TTC-4900 (Türkçe)

7 kategoriden oluşan dengeli bir Türkçe haber verisi. Her kategoride 700 haber, toplam 4.900 örnek.

| Kategori | ID |
|----------|----|
| Siyaset | 0 |
| Ekonomi | 1 |
| Kültür | 2 |
| Sağlık | 3 |
| Spor | 4 |
| Teknoloji | 5 |
| Dünya | 6 |

- **Kaynak:** [Kaggle](https://www.kaggle.com/datasets/savasy/ttc4900) / [HuggingFace](https://huggingface.co/datasets/savasy/ttc4900)

### AG News (İngilizce)

4 kategorili, 120.000 haberlik uluslararası benchmark verisi.

| Kategori | ID |
|----------|----|
| World | 0 |
| Sports | 1 |
| Business | 2 |
| Sci/Tech | 3 |

- **Kaynak:** [HuggingFace](https://huggingface.co/datasets/fancyzhx/ag_news)

---

## Çalıştırma

```bash
# Örnek: LSTM projesini çalıştırma
cd nlp/04-lstm
pip install -r requirements.txt
python lstm_siniflandirma.py
```

Her script kendi `requirements.txt`'sini taşır. Veri `data/` klasöründe aranır; bulunamazsa indirme adresi gösterilir ve sentetik veriyle fallback yapılır. `data/` repoya dahil değildir — herkes kendi indirir.

## Gereksinimler

- Python 3.12+ (gensim uyumsuzluğu nedeniyle 3.14 önerilmez)
- PyTorch, scikit-learn, pandas, matplotlib, seaborn, tqdm, datasets, numpy
