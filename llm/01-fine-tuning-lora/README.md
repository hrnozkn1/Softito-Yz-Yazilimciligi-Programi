# 01 - Fine-tuning + LoRA/QLoRA

GPT-2'ye LoRA ile fine-tuning, QLoRA ile VRAM optimizasyonu.

## Kapsam

- Transformer mimarisi özeti
- Veri hazırlama ve tokenization
- LoRA ile parametre-etkin fine-tuning
- BitsAndBytes 4-bit QLoRA
- Flash-attention, gradient checkpointing
- Model kaydetme, yükleme ve inference

## Kurulum

```bash
cd llm/01-fine-tuning-lora
pip install -r requirements.txt
```

## Çalıştırma

```bash
python fine_tuning_lora.py
```

Script sırayla çalışır: veri yükle → tokenize → LoRA konfigürasyonu → eğitim → kaydet → inference testi.
GPU yoksa CPU'da da çalışır (yavaş).

## Kullanılanlar

`Transformers` · `PEFT` · `BitsAndBytes` · `datasets` · `accelerate`
