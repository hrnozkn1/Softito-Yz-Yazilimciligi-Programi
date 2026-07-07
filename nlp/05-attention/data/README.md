# AG News Veri Seti

Bu klasör, AG News (fancyzhx/ag_news) veri setini içerir.

**Kaynak:** https://huggingface.co/datasets/fancyzhx/ag_news

## İndirme

```python
from datasets import load_dataset
df = load_dataset("fancyzhx/ag_news", split="train").to_pandas()
df.to_csv("ag_news.csv", index=False)
```

## Format

- `ag_news.csv` — CSV, sütunlar: `text`, `label`
- 4 sınıf: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
- 120.000 eğitim, 7.600 test örneği

**Not:** `ag_news.csv` `.gitignore` ile repodan hariç tutulmuştur.
