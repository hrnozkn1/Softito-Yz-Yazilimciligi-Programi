# 06 — Transformer İle Haber Sınıflandırması

**Model:** Transformer Encoder (nn.TransformerEncoder) + Positional Encoding  
**Veri:** AG News — 120.000 haber, 4 kategori

## Mimari

```
Embedding(128) * sqrt(d_model) → PositionalEncoding → TransformerEncoder(2 kat, 4 head) → AdaptiveAvgPool1d → Linear(4)
```

- **Self-attention:** Tüm tokenlar arası ilişkiyi paralel hesaplar
- **Positional Encoding:** Sinüs/kosinüs ile sıra bilgisi
- **Padding mask:** Dolgulu tokenların attention'a katılmasını engeller
- **Adaptive pooling:** Dizi boyutundan bağımsız sabit boyutlu çıktı

## Çalıştırma

```bash
pip install -r requirements.txt
python transformer_siniflandirma.py
```

## Çıktılar

| Dosya | İçerik |
|-------|--------|
| `figures/01_eda.png` | Sınıf dağılımı, token histogramı, en sık kelimeler |
| `figures/02_training_curves.png` | Loss, accuracy, F1 eğrileri |
| `figures/03_confusion_matrix.png` | Karmaşıklık matrisi |

## Hyperparametreler

| Parametre | Değer |
|-----------|-------|
| d_model | 128 |
| nhead | 4 |
| Encoder katman | 2 |
| FF dimension | 256 |
| Dropout | 0.2 |
| Batch | 64 |
| Epoch | 10 |
| Learning rate | 1e-3 |
| MAX_SEQ_LEN | 100 |
| min_freq | 5 |
| Optimizer | Adam + StepLR(step=5, gamma=0.5) |
