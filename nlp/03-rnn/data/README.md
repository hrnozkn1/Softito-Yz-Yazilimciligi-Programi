# TTC-4900 Veri Seti

Bu klasör, `rnn_haber_siniflandirma.py` scriptinin kullandığı TTC-4900 veri setini içerir.

## İndirme

### 1. HuggingFace ile (önerilen)

```python
from datasets import load_dataset
df = load_dataset("savasy/ttc4900", split="train").to_pandas()
df.to_csv("ttc4900.csv", index=False)
```

### 2. Kaggle ile

https://www.kaggle.com/datasets/savasy/ttc4900

## Dosya Formatı

- `ttc4900.csv` — CSV formatında, sütunlar: `category` (0-6), `text`, `category_name`

## Kategoriler

| ID | Kategori |
|----|----------|
| 0 | siyaset |
| 1 | ekonomi |
| 2 | kültür |
| 3 | sağlık |
| 4 | spor |
| 5 | teknoloji |
| 6 | dünya |

**Not:** `ttc4900.csv` dosyası `.gitignore` ile repodan hariç tutulmuştur. Her kullanıcının kendisinin indirmesi gerekir.
