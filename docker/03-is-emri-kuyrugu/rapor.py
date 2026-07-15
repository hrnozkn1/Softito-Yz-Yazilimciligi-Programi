import json
import time
import os
from collections import defaultdict
from datetime import datetime

TAMAMLANAN_KLASOR = "data/tamamlanan"

if __name__ == "__main__":
    print("[rapor] Rapor servisi basladi. 15 saniyede bir ozet basar...\n")
    gorulen = set()

    while True:
        time.sleep(15)

        if not os.path.exists(TAMAMLANAN_KLASOR):
            print("[rapor] tamamlanan klasoru henuz yok, bekleniyor...")
            continue

        dosyalar = [
            d for d in os.listdir(TAMAMLANAN_KLASOR) if d.endswith("_done.json")
        ]

        yeni_dosyalar = [d for d in dosyalar if d not in gorulen]
        for d in dosyalar:
            gorulen.add(d)

        if not yeni_dosyalar and not dosyalar:
            print("[rapor] Henuz tamamlanan islem yok.")
            continue

        tutarlar = []
        sureler = []
        urun_dagilimi = defaultdict(int)
        sehir_dagilimi = defaultdict(int)

        for dosya in dosyalar:
            yol = os.path.join(TAMAMLANAN_KLASOR, dosya)
            try:
                with open(yol, "r", encoding="utf-8") as f:
                    siparis = json.load(f)
                tutarlar.append(siparis["toplam"])
                sureler.append(siparis.get("islem_suresi_sn", 0))
                urun_dagilimi[siparis["urun"]] += 1
                sehir_dagilimi[siparis["sehir"]] += 1
            except Exception:
                continue

        n = len(tutarlar)
        ortalama_tutar = sum(tutarlar) / n if n > 0 else 0
        ortalama_sure = sum(sureler) / n if n > 0 else 0
        en_populer = max(urun_dagilimi, key=urun_dagilimi.get) if urun_dagilimi else "-"

        simdi = datetime.now().strftime("%H:%M:%S")
        print(f"  {'='*45}")
        print(f"  [ {simdi} ]  IS EMRI RAPORU")
        print(f"  {'='*45}")
        print(f"  Islenen siparis    : {n}")
        print(f"  Ortalama tutar     : {ortalama_tutar:,.2f} TL")
        print(f"  Ortalama islem suresi : {ortalama_sure:.2f} saniye")
        print(f"  En populer urun    : {en_populer} ({urun_dagilimi[en_populer]} adet)")
        print(f"  En cok siparis     : {max(sehir_dagilimi, key=sehir_dagilimi.get)} ({max(sehir_dagilimi.values())} siparis)")
        print(f"  {'='*45}\n")
