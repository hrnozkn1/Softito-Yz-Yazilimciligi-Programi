import requests
from bs4 import BeautifulSoup
import re
import os
import csv
from collections import Counter

os.makedirs("cikti", exist_ok=True)

TURKCE_STOP_WORDS = {
    "bir", "ve", "bu", "da", "de", "ile", "için", "olarak", "ama", "çok",
    "daha", "en", "gibi", "ise", "kadar", "ki", "mı", "mi", "mu", "mü",
    "ne", "o", "olan", "olarak", "sonra", "var", "veya", "ya", "yok",
    "çünkü", "şu", "her", "bazı", "diye", "nasıl", "neden", "niye",
    "böyle", "şöyle", "öyle", "artık", "belki", "bile", "gene", "hala",
    "hep", "hiç", "işte", "sadece", "tabii", "yani", "zaten", "zira",
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "is",
}

KAYNAKLAR = [
    ("Hürriyet", "https://www.hurriyet.com.tr/gundem/"),
    ("Milliyet", "https://www.milliyet.com.tr/gundem/"),
    ("Sözcü", "https://www.sozcu.com.tr/"),
]

if __name__ == "__main__":
    tum_basliklar = []

    for site_adi, url in KAYNAKLAR:
        print(f"[{site_adi}] {url} taranıyor...")
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Docker-Egitim/1.0"})
            resp.raise_for_status()
        except Exception as e:
            print(f"  HATA: {e}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        sayac = 0
        for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
            metin = tag.get_text(strip=True)
            if len(metin) < 10 or len(metin) > 200:
                continue
            tum_basliklar.append(f"[{site_adi}] {metin}")
            sayac += 1
        print(f"  {sayac} başlık bulundu")

    if not tum_basliklar:
        print("Hiç başlık bulunamadı. İnternete erişim var mı kontrol et.")
        raise SystemExit(1)

    print(f"\nToplam {len(tum_basliklar)} başlık toplandı.\n")

    with open("cikti/tum_basliklar.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(tum_basliklar))

    tum_kelimeler = []
    for baslik in tum_basliklar:
        baslik = re.sub(r"\[.*?\]", "", baslik)
        baslik = baslik.lower()
        baslik = re.sub(r"[^a-zçğıiöşü ]", "", baslik)
        kelimeler = baslik.split()
        for k in kelimeler:
            if len(k) >= 4 and k not in TURKCE_STOP_WORDS:
                tum_kelimeler.append(k)

    sayac = Counter(tum_kelimeler)
    en_sik_30 = sayac.most_common(30)

    print("En sık 30 kelime:")
    for i, (kelime, adet) in enumerate(en_sik_30, 1):
        print(f"  {i:2}. {kelime:<15} {adet} kez")

    with open("cikti/kelime_sikliklari.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sira", "kelime", "adet"])
        for i, (k, a) in enumerate(en_sik_30, 1):
            writer.writerow([i, k, a])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        kelimeler, adetler = zip(*en_sik_30)

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(list(reversed(kelimeler)), list(reversed(adetler)),
                       color="#2b8cbe", edgecolor="white", height=0.7)
        ax.set_xlabel("Geçiş Sayısı", fontsize=11)
        ax.set_title("Haber Manşetlerinde En Sık Geçen 30 Kelime", fontsize=13, fontweight="bold")
        for bar, val in zip(bars, reversed(adetler)):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig("cikti/kelime_grafigi.png", dpi=120)
        plt.close()
        print("\nGrafik kaydedildi: cikti/kelime_grafigi.png")
    except Exception as e:
        print(f"\nGrafik oluşturulamadı: {e}")

    print("\nTamamlandı.")
