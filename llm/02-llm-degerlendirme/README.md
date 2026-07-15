# 02 - LLM Değerlendirme

BLEU, ROUGE, Perplexity metrikleri ve benchmark runner.

## Kapsam

- Sampling stratejileri (temperature, top-k, top-p)
- BLEU, ROUGE, Perplexity hesaplama
- MMLU / HumanEval benchmark'ları
- LLM-as-Judge (GPT ile puanlama)
- JSON rapor

## Kurulum

```bash
cd llm/02-llm-degerlendirme
pip install -r requirements.txt
```

## Çalıştırma

```bash
python llm_degerlendirme.py
```

## Kullanılanlar

`Transformers` · `datasets` · `evaluate` · `nltk` · `sacrebleu` · `openai`
