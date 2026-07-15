import json
import time
import os
import random
import shutil
from datetime import datetime

os.makedirs("data/tamamlanan", exist_ok=True)

TAMAMLANAN_KLASOR = "data/tamamlanan"
KUYRUK_KLASOR = "data/kuyruk"
HATA_KLASOR = "data/hatali"

if __name__ == "__main__":
    os.makedirs(HATA_KLASOR, exist_ok=True)
    print("[islem] Kuyruk dinleniyor...")

    while True:
        if not os.path.exists(KUYRUK_KLASOR):
            time.sleep(0.5)
            continue

        dosyalar = sorted([
            d for d in os.listdir(KUYRUK_KLASOR) if d.endswith(".json")
        ])

        if not dosyalar:
            time.sleep(0.5)
            continue

        for dosya in dosyalar:
            yol = os.path.join(KUYRUK_KLASOR, dosya)
            try:
                with open(yol, "r", encoding="utf-8") as f:
                    siparis = json.load(f)
            except (json.JSONDecodeError, IOError):
                shutil.move(yol, os.path.join(HATA_KLASOR, dosya))
                continue

            islem_suresi = random.uniform(0.3, 2.0)
            time.sleep(islem_suresi)

            sonuc = {
                **siparis,
                "durum": "tamamlandi",
                "islem_suresi_sn": round(islem_suresi, 2),
                "tamamlanma_zamani": datetime.now().isoformat(),
            }

            hedef = os.path.join(TAMAMLANAN_KLASOR, dosya.replace(".json", "_done.json"))
            with open(hedef, "w", encoding="utf-8") as f:
                json.dump(sonuc, f, ensure_ascii=False)

            os.remove(yol)
            print(f"[islem] ✓ {dosya} islendi "
                  f"({siparis['toplam']:.2f} TL, {islem_suresi:.1f}s)")

        time.sleep(0.5)
