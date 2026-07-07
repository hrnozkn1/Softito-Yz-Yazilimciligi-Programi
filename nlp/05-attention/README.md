# 05 — LSTM + Attention İle Haber Sınıflandırması

**Model:** LSTM (çift yönlü) + Additive (Bahdanau) Attention  
**Veri:** AG News — 120.000 haber, 4 kategori

## Mimari

```
Embedding(128) → BiLSTM(128, 2-layer) → Additive Attention → Dropout → Linear(4)
```

- **Çift yönlü LSTM:** Her token için ileri + geri bağlam
- **Additive Attention:** Tüm LSTM çıktılarına ağırlık verir, son hidden state yerine context vector kullanır
- Attention ağırlıkları görselleştirilir → dil modelinin hangi tokenlara odaklandığı gözlemlenir

## Çalıştırma

```bash
pip install -r requirements.txt
python attention_siniflandirma.py
```

## Çıktılar

| Dosya | İçerik |
|-------|--------|
| `figures/01_eda.png` | Sınıf dağılımı, token histogramı, en sık kelimeler |
| `figures/02_training_curves.png` | Loss, accuracy, F1 eğrileri |
| `figures/03_confusion_matrix.png` | Karmaşıklık matrisi |
| `figures/04_attention_weights.png` | Attention ağırlıklarının token bazında görseli |

## Hyperparametreler

| Parametre | Değer |
|-----------|-------|
| Embedding | 128 |
| Hidden | 128 |
| LSTM katman | 2 (çift yönlü) |
| Batch | 64 |
| Epoch | 10 |
| Learning rate | 1e-3 |
| MAX_SEQ_LEN | 100 |
| min_freq | 5 |
| Optimizer | Adam + StepLR(step=5, gamma=0.5) |
