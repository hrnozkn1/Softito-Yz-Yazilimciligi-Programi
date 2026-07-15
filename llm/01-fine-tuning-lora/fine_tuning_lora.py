"""Fine-tuning + LoRA/QLoRA: Transformer → veri → LoRA → QLoRA → optimizasyon → çıkarım"""
import torch
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

MODEL_ADI = "distilgpt2"
CIKTI_KLASORU = "output"
os.makedirs(CIKTI_KLASORU, exist_ok=True)

ORNEK_VERI = [
    {"soru": "Yapay zeka nedir?", "yanit": "Yapay zeka, makinelerin insan benzeri düşünme ve öğrenme yeteneği kazanmasını sağlayan bilgisayar bilimi dalıdır."},
    {"soru": "Transformer mimarisi ne zaman tanıtıldı?", "yanit": "Transformer mimarisi 2017'de Google tarafından 'Attention Is All You Need' makalesiyle tanıtıldı."},
    {"soru": "LoRA nedir?", "yanit": "LoRA (Low-Rank Adaptation), büyük dil modellerini parametre-etkin şekilde fine-tune etmek için kullanılan bir PEFT yöntemidir. Modelin orijinal ağırlıklarını dondurup küçük adaptör matrisleri ekler."},
    {"soru": "QLoRA'nın farkı nedir?", "yanit": "QLoRA, LoRA'nın 4-bit kuantizasyon ile birleştirilmiş halidir. Temel model 4-bit olarak yüklenir, sadece LoRA adaptörleri normal hassasiyette eğitilir. Bu sayede 7B-70B modeller tek GPU'da fine-tune edilebilir."},
    {"soru": "RAG ne işe yarar?", "yanit": "RAG (Retrieval-Augmented Generation), LLM'lerin harici belgelerden bilgi çekerek daha doğru ve güncel yanıtlar üretmesini sağlayan bir tekniktir."},
    {"soru": "Fine-tuning ile pretraining farkı nedir?", "yanit": "Pretraining, modele genel dil bilgisi kazandırır (aylar sürer, milyonlarca dolar). Fine-tuning ise önceden eğitilmiş modeli küçük, hedefe özel bir veri setiyle ek eğitime tabi tutar (saatler-günler, uygun maliyetli)."},
    {"soru": "Overfitting nedir?", "yanit": "Overfitting, modelin eğitim verisini ezberleyip yeni verilere genelleme yapamaması durumudur. Fine-tuning'de küçük veri setleriyle çalışırken sık karşılaşılır."},
    {"soru": "GPU olmadan fine-tuning yapılabilir mi?", "yanit": "Küçük modellerle (GPT-2, DistilGPT2 gibi) CPU üzerinde fine-tuning yapılabilir, ancak çok yavaştır. Büyük modeller için GPU veya QLoRA ile kuantize edilmiş model ve GPU gereklidir."},
]

def veri_hazirla(ornekler, tokenizer, max_length=256):
    metinler = [f"Soru: {o['soru']}\nYanıt: {o['yanit']}" for o in ornekler]
    tokenized = tokenizer(metinler, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    tokenized["labels"] = tokenized["input_ids"].clone()
    return Dataset.from_dict(tokenized)

def lora_ile_fine_tune(model_adi, veri_listesi, use_4bit=False):
    print(f"Model: {model_adi}")
    tokenizer = AutoTokenizer.from_pretrained(model_adi)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = veri_hazirla(veri_listesi, tokenizer)

    if use_4bit and torch.cuda.is_available():
        print("QLoRA modu: 4-bit kuantizasyon aktif")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_adi, quantization_config=bnb_config, device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print("LoRA modu: tam hassasiyet")
        model = AutoModelForCausalLM.from_pretrained(model_adi)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["c_attn"],
        lora_dropout=0.1, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Eğitilebilir parametre: {trainable:,} / {total:,} (%{100*trainable/total:.2f})")

    training_args = TrainingArguments(
        output_dir=CIKTI_KLASORU, num_train_epochs=10, per_device_train_batch_size=2,
        gradient_accumulation_steps=2, learning_rate=2e-4, weight_decay=0.01,
        logging_steps=5, save_strategy="no", fp16=torch.cuda.is_available(),
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    print("Eğitim başlıyor...")
    trainer.train()

    model.save_pretrained(CIKTI_KLASORU)
    tokenizer.save_pretrained(CIKTI_KLASORU)
    print(f"Model kaydedildi: {CIKTI_KLASORU}/")
    return model, tokenizer

def test_et(model, tokenizer, sorular):
    model.eval()
    device = next(model.parameters()).device
    for soru in sorular:
        prompt = f"Soru: {soru}\nYanıt:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=60, temperature=0.7,
                do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
            )
        yanit = tokenizer.decode(outputs[0], skip_special_tokens=True)
        yanit = yanit.replace(prompt, "").strip()
        print(f"  Soru: {soru}")
        print(f"  Yanıt: {yanit[:150]}")
        print()

def transformer_ozet():
    print("=" * 55)
    print("  TRANSFORMER MİMARİSİ ÖZETİ")
    print("=" * 55)
    print("""
  Transformer (2017, Google): RNN/LSTM'lerin aksine paralel işleme.
  Bileşenler:
    - Self-Attention: Her token'in diğer tüm token'larla ilişkisi
    - Multi-Head Attention: Paralel dikkat mekanizmaları
    - Positional Encoding: Sıra bilgisi (sin/cos)
    - Feed-Forward + Layer Normalization
  Encoder (BERT) → anlama | Decoder (GPT) → üretme
""")

if __name__ == "__main__":
    transformer_ozet()

    model, tokenizer = lora_ile_fine_tune(MODEL_ADI, ORNEK_VERI, use_4bit=False)

    print("\n" + "=" * 55)
    print("  INFERENCE TESTİ")
    print("=" * 55)
    test_sorular = [
        "LoRA ve QLoRA arasındaki fark nedir?",
        "RAG ne işe yarar?",
    ]
    test_et(model, tokenizer, test_sorular)
