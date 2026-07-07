# Türkçe Haber Sınıflandırması — Vanilla RNN

TTC-4900 Türkçe haber veri seti kullanılarak 7 kategorili haber sınıflandırması yapan uçtan uca bir Vanilla RNN projesi. Tek bir Python dosyasında; veriyi yükler, metinleri tokenize eder, keşifsel görseller üretir, **Vanilla RNN** ile model eğitir ve test sonuçlarını raporlar.

> Veri seti: **TTC-4900** · 4.900 haber · 7 kategori · %100 dengeli

## Proje Hakkında

Bu proje, temel RNN mimarisinin metin sınıflandırmasını nasıl yaptığını göstermek üzerine kurulu. Türkçe haber metinleri karakter bazlı tokenize edilir, sözlük oluşturulur, diziler pad'lenir ve Vanilla RNN ile sınıflandırılır.

Kategoriler: siyaset, ekonomi, kültür, sağlık, spor, teknoloji, dünya

## Model Mimarisi

Embedding(128) → nn.RNN(hidden=128, 1 katman, tanh) → Dropout(0.3) → Linear(7) → CrossEntropyLoss

## Hiperparametreler

| Parametre | Değer |
|-----------|-------|
| Kelime dağarcığı | 28.828 |
| Maks dizi uzunluğu | 50 token |
| Embedding boyutu | 128 |
| Gizli durum boyutu | 128 |
| RNN katmanı | 1 (Vanilla, tanh) |
| Dropout | 0.3 |
| Öğrenme oranı | 2e-3 |
| Batch boyutu | 32 |
| Epoch | 20 |

## Veri Seti

**TTC-4900** (savasy/ttc4900):
- 4.900 Türkçe haber metni
- 7 kategori (700'er örnek, tam dengeli)
- Kaynak: Türkçe haber siteleri
- Kaggle: https://www.kaggle.com/datasets/savasy/ttc4900
- HuggingFace: https://huggingface.co/datasets/savasy/ttc4900

## Sonuçlar

| Metrik | Değer |
|--------|-------|
| Accuracy | **%41.0** |
| F1 (macro) | **0.413** |

| Kategori | Precision | Recall | F1 |
|----------|-----------|--------|----|
| siyaset | 0.32 | 0.49 | 0.38 |
| ekonomi | 0.37 | 0.35 | 0.36 |
| kültür | 0.29 | 0.31 | 0.30 |
| sağlık | 0.53 | 0.44 | 0.48 |
| spor | **0.60** | **0.58** | **0.59** |
| teknoloji | 0.52 | 0.35 | 0.42 |
| dünya | 0.36 | 0.34 | 0.35 |

> **Not:** Vanilla RNN uzun metinlerde (ortalama 270 token) vanishing gradient sorunu yaşar. Bu projede ilk 50 token kullanılarak %41 doğruluk elde edilmiştir (rastgele tahmin: %14.3). Spor ve sağlık kategorileri kendine özgü kelime dağarcığı sayesinde en yüksek başarıyı gösterirken, kültür kategorisi diğerleriyle örtüşen kelimeler nedeniyle en düşük F1 skoruna sahiptir.

## Görseller

| Görsel | Açıklama |
|--------|----------|
| `figures/01_eda.png` | Sınıf dağılımı, dizi uzunluğu, kelime frekansı |
| `figures/02_training_curves.png` | Epoch bazında kayıp ve doğruluk/F1 |
| `figures/03_confusion_matrix.png` | Karmaşıklık matrisi |

## Çalıştırma

```bash
pip install -r requirements.txt
```

Veriyi indirip `data/ttc4900.csv` olarak kaydedin:
```python
# HuggingFace ile
from datasets import load_dataset
df = load_dataset("savasy/ttc4900", split="train").to_pandas()
df.to_csv("data/ttc4900.csv", index=False)
```

Veya doğrudan Kaggle'dan: https://www.kaggle.com/datasets/savasy/ttc4900

Sonra çalıştırın:
```bash
python rnn_haber_siniflandirma.py
```

Script, `data/ttc4900.csv` bulamazsa otomatik olarak sentetik fallback verisi oluşturur ve onunla çalışır.

## Proje Yapısı

```
nlp/03-rnn/
├── rnn_haber_siniflandirma.py   # Ana Python scripti
├── README.md                    # Bu dosya
├── requirements.txt             # Bağımlılıklar
├── .gitignore                   # data/ hariç
├── data/
│   └── ttc4900.csv              # TTC-4900 veri seti (manuel indirilir)
└── figures/                     # Otomatik oluşturulan görseller
```

## Kaynaklar

- Yıldırım, S., & Yıldız, T. (2018). A comparative analysis of text classification for Turkish language. *Pamukkale University Journal of Engineering Sciences*, 24(5), 879-886.
- TTC-4900: https://www.kaggle.com/datasets/savasy/ttc4900
