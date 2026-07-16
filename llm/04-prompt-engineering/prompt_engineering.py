"""Prompt Engineering: zero-shot, few-shot, CoT, system prompt"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def get_client():
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit("OPENAI_API_KEY gerekli (.env dosyasına ekle)")
    return OpenAI(api_key=key)

def ask(client, system, user, **kwargs):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    resp = client.chat.completions.create(
        model="gpt-4o-mini", messages=msgs, temperature=kwargs.get("temperature", 0), **{k: v for k, v in kwargs.items() if k != "temperature"},
    )
    return resp.choices[0].message.content

def zero_shot(client):
    print("=" * 50)
    print("  ZERO-SHOT")
    print("=" * 50)
    yanit = ask(client, None, "Python'da bir listenin eleman sayısını nasıl bulursun?")
    print(f"  Yanıt: {yanit[:150]}...\n")

def few_shot(client):
    print("=" * 50)
    print("  FEW-SHOT (3 örnek)")
    print("=" * 50)
    prompt = """Soru: 5 elma + 3 elma = ?
Yanıt: 8 elma

Soru: 10 kitap - 4 kitap = ?
Yanıt: 6 kitap

Soru: 7 kalem × 2 = ?
Yanıt: 14 kalem

Soru: 20 lira ÷ 4 kişi = ?
Yanıt:"""
    yanit = ask(client, None, prompt)
    print(f"  Yanıt: {yanit.strip()}\n")

def chain_of_thought(client):
    print("=" * 50)
    print("  CHAIN-OF-THOUGHT (CoT)")
    print("=" * 50)
    prompt = """Bir sınıfta 15 kız, 12 erkek öğrenci var. 3 kız ve 2 erkek sınıftan ayrılıyor.
Kalan öğrencilerin yüzde kaçı kızdır? Adım adım düşün."""
    yanit = ask(client, None, prompt)
    print(f"  Yanıt: {yanit[:200]}...\n")

def system_prompt_demo(client):
    print("=" * 50)
    print("  SYSTEM PROMPT")
    print("=" * 50)

    prompts = {
        "Yardımsever asistan": "Sen yardımsever bir asistansın. Her soruya nazik ve detaylı yanıt ver.",
        "Kısa ve öz": "Her soruya tek cümleyle, en fazla 15 kelimeyle yanıt ver.",
        "Şair": "Tüm yanıtlarını kafiyeli şiir şeklinde yaz.",
    }

    for isim, sp in prompts.items():
        yanit = ask(client, sp, "Python nedir?")
        print(f"  [{isim}]: {yanit[:100]}...\n")

def temperature_demo(client):
    print("=" * 50)
    print("  TEMPERATURE ETKİSİ")
    print("=" * 50)

    for t in [0.0, 0.5, 1.0, 1.5]:
        yanit = ask(client, None, "Bana kısa bir hikaye anlat.", temperature=t)
        print(f"  T={t:.1f}: {yanit[:80]}...")
    print()

if __name__ == "__main__":
    client = get_client()
    zero_shot(client)
    few_shot(client)
    chain_of_thought(client)
    system_prompt_demo(client)
    temperature_demo(client)
