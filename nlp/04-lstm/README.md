# 04 — LSTM İle Haber Sınıflandırması

**Model:** LSTM (2 katman, 128 hidden, dropout 0.3)  
**Veri:** AG News — 120.000 haber, 4 kategori (World/Sports/Business/Sci-Tech)

## Mimari

```
Embedding(128) → LSTM(128, 2-layer) → Dropout → Linear(4)
```

- LSTM, Vanilla RNN'deki vanishing gradient sorununu çözer
- Çift yönlü değil, tek yönlü — son hidden state'i sınıflandırıcıya verir
- Gradient clipping ile eğitim stabilitesi

## Çalıştırma

```bash
pip install -r requirements.txt
python lstm_siniflandirma.py
```

Script otomatik olarak `data/ag_news.csv`'yi yükler; bulamazsa HuggingFace'den indirme talimatı gösterir.

## Çıktılar

| Dosya | İçerik |
|-------|--------|
| `figures/01_eda.png` | Sınıf dağılımı, token histogramı, en sık kelimeler |
| `figures/02_training_curves.png` | Loss, accuracy, F1 eğrileri |
| `figures/03_confusion_matrix.png` | Karmaşıklık matrisi |

## Hyperparametreler

| Parametre | Değer |
|-----------|-------|
| Embedding | 128 |
| Hidden | 128 |
| LSTM katman | 2 |
| Batch | 64 |
| Epoch | 10 |
| Learning rate | 1e-3 |
| MAX_SEQ_LEN | 100 |
| min_freq | 5 |
| Optimizer | Adam + StepLR(step=5, gamma=0.5) |
