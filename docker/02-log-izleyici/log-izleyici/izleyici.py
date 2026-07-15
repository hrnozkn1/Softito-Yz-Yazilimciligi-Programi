import docker
import time
import re
from collections import defaultdict
from datetime import datetime

client = docker.from_env()

HATA_KODLARI = {"401", "403", "404", "500", "502", "503"}
BASARI_KODU = "200"

KALIPLAR = [
    (r"\b4\d{2}\b", "4XX (istemci hatasi)"),
    (r"\b5\d{2}\b", "5XX (sunucu hatasi)"),
    (r"\btimeout\b", "TIMEOUT"),
    (r"\bconnection refused\b", "BAGLANTI_REDDI"),
]

def ayrintili_tara(log_satiri):
    bulunan = []
    for kalip, etiket in KALIPLAR:
        if re.search(kalip, log_satiri, re.IGNORECASE):
            bulunan.append(etiket)
    return bulunan

if __name__ == "__main__":
    print("[izleyici] Log izleme basladi. uretici konteyneri bekleniyor...")
    time.sleep(3)

    container = None
    for _ in range(30):
        try:
            container = client.containers.get("uretici")
            break
        except Exception:
            time.sleep(1)
    else:
        print("[izleyici] HATA: 'uretici' konteyneri bulunamadi.")
        raise SystemExit(1)

    print("[izleyici] uretici bulundu, loglar okunuyor...\n")

    log_akisi = container.logs(stream=True, follow=True, tail=0, timestamps=True)

    toplam = hata_sayisi = 0
    son_rapor_ani = time.time()

    for ham_satir in log_akisi:
        satir = ham_satir.decode("utf-8", errors="replace").strip()
        if not satir:
            continue

        toplam += 1
        hatali_mi = any(kod in satir for kod in HATA_KODLARI)
        if hatali_mi:
            hata_sayisi += 1
            bulunan = ayrintili_tara(satir)
            if bulunan:
                print(f"  [!] {', '.join(bulunan)} → {satir[:100]}")

        if time.time() - son_rapor_ani >= 10:
            hata_orani = (hata_sayisi / toplam * 100) if toplam > 0 else 0
            simdi = datetime.now().strftime("%H:%M:%S")
            print(f"\n{'='*50}")
            print(f"  [{simdi}] RAPOR")
            print(f"  Toplam istek : {toplam}")
            print(f"  Hatali istek : {hata_sayisi}")
            print(f"  Hata orani   : %{hata_orani:.1f}")
            print(f"{'='*50}\n")
            son_rapor_ani = time.time()
