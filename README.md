# Softito YZ Yazılımcılığı Programı

NLP (Doğal Dil İşleme), ML (Makine Öğrenmesi) ve Docker ile konteynerizasyon alanlarında sıfırdan ileri seviyeye uzanan projeler. Her proje bağımsız çalışır: çalıştır, eğit, grafikleri gör, karşılaştır.

## NLP Projeleri

| # | Proje | Ne Öğretir? | Veri Seti | Model |
|---|-------|-------------|-----------|-------|
| 01 | [TF-IDF](nlp/01-tf-idf/) | Kelimeleri sayıya dökme, vektör uzayı | [TTC-4900](https://www.kaggle.com/datasets/savasy/ttc4900) (Türkçe haber, 7 kategori) | TF-IDF + LogisticRegression |
| 02 | [Word Embeddings](nlp/02-word-embeddings/) | Anlamsal gömmeler, kelime benzerliği | TTC-4900 | Word2Vec, FastText, TF-IDF + t-SNE |
| 03 | [Vanilla RNN](nlp/03-rnn/) | Tekrarlayan sinir ağları, sequence modelleme | TTC-4900 | `nn.RNN` (PyTorch) — %41 accuracy |
| 04 | [LSTM](nlp/04-lstm/) | Vanishing gradient sorununun çözümü | [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) (İngilizce, 4 kategori, 120K haber) | `nn.LSTM` |
| 05 | [LSTM + Attention](nlp/05-attention/) | Modelin hangi kelimelere odaklandığını görmek | AG News | BiLSTM + Bahdanau Attention |
| 06 | [Transformer](nlp/06-transformer/) | Self-attention, paralel işleme, sıra bağımsızlık | AG News | `nn.TransformerEncoder` |

## ML Projeleri

| # | Proje | Ne Öğretir? | Veri Seti | Model/Yöntem |
|---|-------|-------------|-----------|-------------|
| 01 | [Futbolcu Kümeleme](ml/01-futbolcu-kumeleme/) | Denetimsiz öğrenme, kümeleme analizi | Futbolcu verileri | K-Means, Hiyerarşik Kümeleme, GMM, PCA |
| 02 | [Hava Durumu RNN](ml/02-hava-durumu-rnn/) | Zaman serisi tahmini, RNN | Open-Meteo API (İstanbul sıcaklık) | `nn.RNN` (PyTorch) |
| 03 | [Kalp Hastalığı Tahmini](ml/03-kalp-hastaligi-tahmin/) | Sınıflandırma modelleri karşılaştırması | Kalp hastalığı verisi | Logistic Regression, SVM |
| 04 | [Araba Fiyat Regresyonu](ml/04-araba-fiyat-regresyon/) | Regresyon analizi, aykırı değer temizliği | Car Dekho | Simple/Multiple Linear Regression |
| 05 | [Spotify Churn EDA](ml/05-spotify-churn-eda/) | Keşifçi veri analizi (EDA) | Spotify müşteri verisi | EDA (Pandas, Seaborn) |
| 06 | [Su Kalitesi Sınıflandırması](ml/06-su-kalitesi-siniflandirma/) | KNN ve Naive Bayes karşılaştırması | Water Potability | KNN, Gaussian Naive Bayes |
| 07 | [Telefon Fiyat Sınıflandırması](ml/07-telefon-fiyat-siniflandirma/) | Çok sınıflı sınıflandırma | Telefon özellikleri | Decision Tree, Random Forest |
| 08 | [Telekom Churn XGBoost](ml/08-telekom-churn-xgboost/) | Müşteri kaybı tahmini, hiperparametre optimizasyonu | Telco müşteri verisi | AdaBoost, XGBoost, GridSearchCV |
| 09 | [VADER Duygu Analizi](ml/09-vader-duygu-analizi/) | Kural tabanlı duygu analizi | IMDB Film Yorumları | VADER (NLTK) |

### Mimari Gelişimi

```
TF-IDF ──→ Word2Vec ──→ RNN ──→ LSTM ──→ LSTM+Attn ──→ Transformer
(1990s)    (2013)      (1986)   (1997)    (2015)        (2017)
```

### Kullanılan Veri Setleri

| Veri Seti | Alan | Sınıflar | Örnek Sayısı | Kullanıldığı Projeler |
|-----------|------|----------|-------------|----------------------|
| [TTC-4900](https://www.kaggle.com/datasets/savasy/ttc4900) | NLP 🇹🇷 | siyaset, ekonomi, kültür, sağlık, spor, teknoloji, dünya | 4.900 | NLP 01, 02, 03 |
| [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) | NLP 🇬🇧 | World, Sports, Business, Sci/Tech | 120.000 | NLP 04, 05, 06 |
| Car Dekho | ML 🚗 | - (regresyon) | ~4.000 | ML 04 |
| Spotify Müşteri | ML 🎵 | Churn/Not Churn | 8.000 | ML 05 |
| Water Potability | ML 💧 | Potable/Not Potable | ~3.200 | ML 06 |
| Telefon Özellikleri | ML 📱 | 4 fiyat segmenti | 2.000 | ML 07 |
| Telco Müşteri | ML 📞 | Churn/Not Churn | ~7.000 | ML 08 |
| IMDB Film Yorumları | ML 🎬 | Pozitif/Negatif | 40.000 | ML 09 |

TTC-4900 dengeli bir Türkçe haber verisi (her sınıftan 700 haber). AG News ise uluslararası bir benchmark. ML projeleri ise çeşitli alanlardan gerçek dünya verileriyle çalışır.

## Docker Projeleri

> 🐳 Docker pratikleri ayrı bir repoda: **[softito-docker-pratikleri](https://github.com/hrnozkn1/softito-docker-pratikleri)**

Tek konteynerden çok servisli mimariye, basitten karmaşığa ilerleyen 3 proje:

| # | Proje | Konu | Docker Komutu | Seviye |
|---|-------|------|---------------|--------|
| 01 | [Web Kazıyıcı](https://github.com/hrnozkn1/softito-docker-pratikleri/tree/main/01-web-kaziyici) | Dockerfile · Volume mount · Konteyner yaşam döngüsü | `docker build` / `docker run` | ⭐ Giriş |
| 02 | [Log İzleyici](https://github.com/hrnozkn1/softito-docker-pratikleri/tree/main/02-log-izleyici) | Docker Compose · Servis ağı · Container logs | `docker compose up` | ⭐⭐ Orta |
| 03 | [İş Emri Kuyruğu](https://github.com/hrnozkn1/softito-docker-pratikleri/tree/main/03-is-emri-kuyrugu) | Docker Compose · Shared volume · Producer-consumer | `docker compose up` | ⭐⭐ Orta |

**Not:** Bu projelerde ML/NLP yoktur. Amaç Docker'ın temel yapı taşlarını (Dockerfile, Compose, volume, network) kavramaktır.

## Nasıl Çalıştırılır

### NLP Projeleri (Python script)
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

### ML Projeleri (Jupyter Notebook)
```bash
cd Softito-Yz-Yazilimciligi-Programi/ml/04-araba-fiyat-regresyon
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook
```

Notebook'u açıp hücreleri sırayla çalıştırın. Her notebook bağımsızdır, veri setleri kendi klasöründe bulunur.

### Klasör Yapısı

```
├── nlp/XX-proje/
│   ├── README.md
│   ├── requirements.txt
│   ├── .gitignore              # data/ hariç (gitignored)
│   ├── data/README.md          # veri seti bilgisi
│   ├── figures/                # çıktı grafikleri
│   └── proje_adi.py            # tek dosya, çalıştır ve gör
│
├── ml/XX-proje/
│   ├── README.md
│   ├── proje_adi.ipynb          # Jupyter Notebook
│   └── *.csv                    # veri setleri
│
└── docker/                     # Ayrı repo: softito-docker-pratikleri
    ├── 01-web-kaziyici/        # Tek konteyner
    ├── 02-log-izleyici/        # Docker Compose, 2 servis
    └── 03-is-emri-kuyrugu/     # Docker Compose, 3 servis + volume
```

## Gereksinimler

- Python 3.10+ önerilir
- PyTorch, scikit-learn, pandas, matplotlib, seaborn, tqdm, datasets, numpy, nltk
- Jupyter Notebook (ML projeleri için)

## Notlar

- NLP: 03-rnn (Vanilla RNN) test edildi ve sonuçları push edildi — %41 accuracy
- NLP: 04-05-06 scriptleri hazır ancak eğitim tam sonuç için ~15-20 dk sürer (CPU'da çalıştırılmalı)
- GPU varsa `DEVICE` otomatik algılanır, eğitim çok daha hızlı olur
- ML: Tüm projeler Jupyter Notebook formatındadır, her biri bağımsız çalışır
