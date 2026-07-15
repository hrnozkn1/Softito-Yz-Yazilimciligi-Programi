"""LLM Değerlendirme: sampling, metrikler, benchmark, LLM-as-Judge, rapor"""
import torch
import json
import os
import math
from datetime import datetime
from collections import Counter

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import evaluate

CIKTI_KLASORU = "output"
os.makedirs(CIKTI_KLASORU, exist_ok=True)

def sampling_demo():
    print("=" * 55)
    print("  SAMPLING STRATEJİLERİ")
    print("=" * 55)

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    prompt = "Yapay zeka"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    stratejiler = [
        ("Greedy", dict(do_sample=False, max_new_tokens=20)),
        ("Temperature=0.7", dict(do_sample=True, temperature=0.7, max_new_tokens=20)),
        ("Temperature=1.5", dict(do_sample=True, temperature=1.5, max_new_tokens=20)),
        ("Top-k=10", dict(do_sample=True, top_k=10, temperature=0.7, max_new_tokens=20)),
        ("Top-p=0.9", dict(do_sample=True, top_p=0.9, temperature=0.7, max_new_tokens=20)),
    ]

    print(f"\nPrompt: '{prompt}'\n")
    for isim, kwargs in stratejiler:
        with torch.no_grad():
            out = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, **kwargs)
        metin = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"  {isim:<18}: {metin}")

    return model, tokenizer

def perplexity_hesapla(model, tokenizer):
    print("\n" + "=" * 55)
    print("  PERPLEXITY")
    print("=" * 55)

    test_metinleri = [
        "Yapay zeka, insan benzeri düşünme yeteneği kazandırmayı amaçlayan bir bilgisayar bilimi dalıdır.",
        "Bugün hava çok güzel, parka yürüyüşe gideceğim.",
        "Transformer mimarisi self-attention mekanizması ile paralel işleme sağlar.",
        "asdfg hjkl qwert zuıop cvbnm wxyz 123456",
    ]

    for metin in test_metinleri:
        inputs = tokenizer(metin, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        ppl = math.exp(outputs.loss.item())
        print(f"  PPL={ppl:8.1f} | {metin[:60]}...")

def bleu_hesapla():
    print("\n" + "=" * 55)
    print("  BLEU")
    print("=" * 55)

    referans = "Transformer mimarisi 2017'de Google tarafından geliştirildi."
    tahminler = [
        "Transformer Google tarafından 2017'de geliştirilmiş bir mimaridir.",
        "Transformer 2017'de Google tarafından geliştirilmiştir.",
        "Google 2017'de ortaya çıkardı Transformer'ı.",
        "Bugün hava çok güzel ve güneşli.",
    ]

    try:
        bleu = evaluate.load("bleu")
        for i, t in enumerate(tahminler):
            sonuc = bleu.compute(predictions=[t], references=[[referans]])
            print(f"  [{i+1}] BLEU={sonuc['bleu']:.4f} | {t[:70]}...")
    except Exception as e:
        print(f"  BLEU yüklenemedi: {e}")

def rouge_hesapla():
    print("\n" + "=" * 55)
    print("  ROUGE")
    print("=" * 55)

    referans = "Yapay zeka makinelerin insan gibi düşünmesini ve öğrenmesini sağlayan bir teknolojidir."
    tahmin = "Yapay zeka insanlar gibi düşünebilen makineler yaratmayı hedefleyen bir teknolojidir."

    try:
        rouge = evaluate.load("rouge")
        sonuc = rouge.compute(predictions=[tahmin], references=[referans])
        for k, v in sonuc.items():
            print(f"  {k}: {v:.4f}")
    except Exception as e:
        print(f"  ROUGE yüklenemedi: {e}")

def llm_as_judge_demo():
    print("\n" + "=" * 55)
    print("  LLM-AS-JUDGE")
    print("=" * 55)
    print("  (OpenAI API anahtarı varsa çalışır, yoksa demo modu)")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("  API anahtarı yok, demo atlanıyor.")
        return

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        soru = "LoRA nedir?"
        yanit_a = "LoRA, büyük dil modellerini daha az parametreyle eğitmek için geliştirilmiş bir yöntemdir."
        yanit_b = "LoRA bir tür yapay zeka modelidir ve çok karmaşıktır."

        prompt = f"""Aşağıdaki iki yanıtı değerlendir. Hangisi daha doğru ve açıklayıcı?
Soru: {soru}
Yanıt A: {yanit_a}
Yanıt B: {yanit_b}
Sadece 'A', 'B' veya 'Eşit' yaz."""

        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=5,
        )
        print(f"  Kazanan: {resp.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"  Hata: {e}")

def rapor_kaydet(metrikler):
    dosya = os.path.join(CIKTI_KLASORU, f"rapor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(dosya, "w", encoding="utf-8") as f:
        json.dump(metrikler, f, ensure_ascii=False, indent=2)
    print(f"\nRapor kaydedildi: {dosya}")

if __name__ == "__main__":
    print("LLM DEĞERLENDİRME\n")

    model, tokenizer = sampling_demo()
    perplexity_hesapla(model, tokenizer)
    bleu_hesapla()
    rouge_hesapla()
    llm_as_judge_demo()

    rapor_kaydet({
        "tarih": datetime.now().isoformat(),
        "metrikler": ["perplexity", "bleu", "rouge", "llm_as_judge"],
        "not": "Tüm skorlar demo amaçlıdır.",
    })
