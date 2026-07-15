import json
import time
import random
import os
from datetime import datetime

os.makedirs("data/kuyruk", exist_ok=True)
os.makedirs("data/tamamlanan", exist_ok=True)

URUNLER = [
    ("Klavye", 450), ("Mouse", 320), ("Monitör", 3200),
    ("Kulaklık", 580), ("Webcam", 750), ("USB Bellek", 180),
    ("HDMI Kablo", 120), ("Adaptör", 250), ("Mousepad", 90),
    ("Laptop Standı", 420), ("Tablet", 4500), ("Powerbank", 390),
]
SEHIRLER = ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"]

if __name__ == "__main__":
    print("[uretim] Uretim basladi. 2 saniyede bir siparis uretiyor...")
    siparis_no = 0
    while True:
        urun, fiyat = random.choice(URUNLER)
        adet = random.randint(1, 3)
        siparis = {
            "siparis_no": siparis_no,
            "urun": urun,
            "birim_fiyat": fiyat,
            "adet": adet,
            "toplam": round(fiyat * adet * random.uniform(0.9, 1.1), 2),
            "sehir": random.choice(SEHIRLER),
            "zaman": datetime.now().isoformat(),
        }
        dosya_adi = f"data/kuyruk/siparis_{siparis_no:06d}.json"
        with open(dosya_adi, "w", encoding="utf-8") as f:
            json.dump(siparis, f, ensure_ascii=False)
        print(f"[uretim] Siparis #{siparis_no} → {dosya_adi}")
        siparis_no += 1
        time.sleep(2)
