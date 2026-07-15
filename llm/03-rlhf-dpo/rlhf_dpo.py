"""RLHF + DPO: Reward model → PPO → DPO → karşılaştırma"""
import torch
import os
from copy import deepcopy

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

CIKTI_KLASORU = "output"
os.makedirs(CIKTI_KLASORU, exist_ok=True)

MODEL_ADI = "distilgpt2"

TERCIH_VERISI = [
    {
        "prompt": "Yapay zeka nedir?",
        "chosen": "Yapay zeka, makinelerin insan benzeri bilişsel yetenekler kazanmasını sağlayan bilgisayar bilimi dalıdır.",
        "rejected": "Yapay zeka bilgisayarlarla ilgili bir şeydir sanırım tam bilmiyorum.",
    },
    {
        "prompt": "Python'da liste nasıl oluşturulur?",
        "chosen": "Python'da liste oluşturmak için köşeli parantez kullanılır: liste = [1, 2, 3]",
        "rejected": "liste yapmak için şey yaparsın işte yazarsın olur.",
    },
    {
        "prompt": "Güneş sistemi kaç gezegenden oluşur?",
        "chosen": "Güneş sistemi 8 gezegenden oluşur: Merkür, Venüs, Dünya, Mars, Jüpiter, Satürn, Uranüs ve Neptün.",
        "rejected": "Güneş sistemi 9 gezegen aslında Plüton da var mı yok mu karışık.",
    },
    {
        "prompt": "Docker ne işe yarar?",
        "chosen": "Docker, uygulamaları konteyner adı verilen izole ortamlarda paketleyip çalıştırmayı sağlayan bir platformdur.",
        "rejected": "Docker işte yazılım falan çalıştırmak için bir araç.",
    },
    {
        "prompt": "LoRA ve QLoRA farkı nedir?",
        "chosen": "LoRA, modele küçük adaptör matrisleri ekleyerek parametre-etkin fine-tuning yapar. QLoRA ise temel modeli 4-bit kuantize edip LoRA'yı bunun üzerine uygular, böylece VRAM kullanımı dramatik düşer.",
        "rejected": "LoRA ve QLoRA benzer şeyler, QLoRA biraz daha iyi galiba.",
    },
    {
        "prompt": "Transformer mimarisinde self-attention ne yapar?",
        "chosen": "Self-attention, dizideki her token'in diğer tüm token'larla olan ilişkisini ağırlıklı olarak hesaplar. Bu sayede model uzak bağımlılıkları yakalayabilir.",
        "rejected": "Self-attention token'ları birbirine bağlar işte.",
    },
    {
        "prompt": "Overfitting nasıl önlenir?",
        "chosen": "Overfitting'i önlemek için dropout, early stopping, data augmentation, regularization (L1/L2) ve cross-validation gibi teknikler kullanılabilir.",
        "rejected": "Overfitting olursa daha az eğitirsin.",
    },
    {
        "prompt": "Gradient descent nedir?",
        "chosen": "Gradient descent, bir kayıp fonksiyonunu minimize etmek için parametreleri gradyanın tersi yönünde adım adım güncelleyen bir optimizasyon algoritmasıdır.",
        "rejected": "Gradient descent optimizasyon için kullanılır, modeli iyileştirir.",
    },
]

def reward_model_demo():
    print("=" * 55)
    print("  REWARD MODEL")
    print("=" * 55)
    print("""
  Amaç: İnsan tercihlerini sayısal ödül sinyaline dönüştürmek.

  Akış:
    1. İnsanlar model çıktılarını karşılaştırır (A > B)
    2. Tercih çiftleri (chosen, rejected) toplanır
    3. Küçük bir model (reward model) bu çiftlerle eğitilir
    4. Reward model, her yanıta 0-1 arası ödül puanı verir

  Bu projede {len(TERCIH_VERISI)} tercih çifti kullanılıyor (demo amaçlı).
  Gerçek RLHF'te binlerce çift gerekir.
""")

def dpo_egitimi():
    print("=" * 55)
    print("  DPO (Direct Preference Optimization)")
    print("=" * 55)
    print("""
  DPO, RLHF'in daha sade bir alternatifidir.
  PPO'dan farkı: ayrı bir reward model eğitmeye gerek yoktur.
  Tercih verisi doğrudan loss fonksiyonuna dahil edilir.

  DPO Loss:
    L = -log(sigma(beta * (log_pi(chosen) - log_pi(rejected))))

  pi: eğitilen model, beta: sıcaklık parametresi
  Amaç: chosen'ın olasılığını artır, rejected'ın olasılığını azalt.
""")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ADI)
    tokenizer.pad_token = tokenizer.eos_token

    train_data = []
    for ornek in TERCIH_VERISI:
        train_data.append({
            "prompt": ornek["prompt"],
            "chosen": ornek["chosen"],
            "rejected": ornek["rejected"],
        })

    try:
        from trl import DPOTrainer, DPOConfig
        from peft import LoraConfig, get_peft_model, TaskType

        print("DPO eğitimi başlıyor...")

        model = AutoModelForCausalLM.from_pretrained(MODEL_ADI)
        model_ref = AutoModelForCausalLM.from_pretrained(MODEL_ADI)

        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["c_attn"],
            lora_dropout=0.1, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model_ref = get_peft_model(model_ref, lora_config)

        train_dataset = Dataset.from_list(train_data)

        dpo_config = DPOConfig(
            output_dir=CIKTI_KLASORU,
            num_train_epochs=5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            logging_steps=5,
            save_strategy="no",
            beta=0.1,
            max_length=256,
            max_prompt_length=128,
        )

        trainer = DPOTrainer(
            model=model, ref_model=model_ref, args=dpo_config,
            train_dataset=train_dataset, processing_class=tokenizer,
        )
        trainer.train()

        model.save_pretrained(CIKTI_KLASORU)
        tokenizer.save_pretrained(CIKTI_KLASORU)
        print(f"DPO model kaydedildi: {CIKTI_KLASORU}/")
        return tokenizer, True

    except ImportError:
        print("  trl kütüphanesi yüklü değil, DPO demo modunda.")
        print("  Kurmak için: pip install trl")
        return tokenizer, False
    except Exception as e:
        print(f"  DPO eğitimi atlandı: {e}")
        return tokenizer, False

def karsilastirma(tokenizer, dpo_var=False):
    print("\n" + "=" * 55)
    print("  KARŞILAŞTIRMA: Base vs DPO")
    print("=" * 55)

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ADI)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = base_model.to(device)
    base_model.eval()

    test_prompts = [
        "Yapay zeka nedir?",
        "Python'da liste nasıl oluşturulur?",
        "Docker ne işe yarar?",
    ]

    for prompt in test_prompts:
        inputs = tokenizer(f"Soru: {prompt}\nYanıt:", return_tensors="pt").to(device)
        with torch.no_grad():
            out_base = base_model.generate(
                **inputs, max_new_tokens=50, temperature=0.7, do_sample=True,
                top_p=0.9, pad_token_id=tokenizer.eos_token_id,
            )
        yanit_base = tokenizer.decode(out_base[0], skip_special_tokens=True)
        yanit_base = yanit_base.split("Yanıt:")[-1].strip()

        print(f"\n  Prompt: {prompt}")
        print(f"  Base : {yanit_base[:150]}")

        if dpo_var:
            dpo_model = AutoModelForCausalLM.from_pretrained(CIKTI_KLASORU).to(device)
            dpo_model.eval()
            with torch.no_grad():
                out_dpo = dpo_model.generate(
                    **inputs, max_new_tokens=50, temperature=0.7, do_sample=True,
                    top_p=0.9, pad_token_id=tokenizer.eos_token_id,
                )
            yanit_dpo = tokenizer.decode(out_dpo[0], skip_special_tokens=True)
            yanit_dpo = yanit_dpo.split("Yanıt:")[-1].strip()
            print(f"  DPO  : {yanit_dpo[:150]}")

def rlhf_akisi():
    print("=" * 55)
    print("  RLHF AKIŞI (PPO)")
    print("=" * 55)
    print("""
  RLHF (Reinforcement Learning from Human Feedback) 3 aşamalıdır:

  1. SFT (Supervised Fine-Tuning): Base model, kaliteli yanıtlarla eğitilir.
  2. Reward Model: İnsan tercihleriyle ödül modeli eğitilir.
  3. PPO: Model, reward model'den gelen sinyalle pekiştirmeli öğrenme yapar.

  PPO (Proximal Policy Optimization):
    - Model çıktı üretir → reward model puanlar → model güncellenir
    - KL divergence cezası: modelin orijinal davranıştan çok sapmasını önler
    - PPO-clip: güncelleme adımını sınırlar, stabil eğitim sağlar

  DPO vs RLHF:
    DPO → Tek aşamalı, reward model yok, daha basit, daha stabil.
    RLHF → 3 aşamalı, reward model var, daha esnek, endüstri standardı.
""")

if __name__ == "__main__":
    print("RLHF + DPO\n")
    rlhf_akisi()
    reward_model_demo()
    tokenizer, dpo_var = dpo_egitimi()
    karsilastirma(tokenizer, dpo_var)
