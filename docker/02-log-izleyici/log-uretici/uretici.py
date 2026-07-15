import time
import random
from datetime import datetime, timedelta

IPLER = [f"192.168.1.{i}" for i in range(1, 21)]
URL_LER = ["/", "/urun/42", "/urun/99", "/hakkimizda", "/iletisim",
           "/giris", "/kayit", "/sepet", "/siparis/onay", "/admin",
           "/api/v1/products", "/api/v1/users", "/eski-sayfa", "/resimler/logo.png"]
METOTLAR = ["GET", "GET", "GET", "GET", "GET", "POST", "POST", "DELETE", "PUT"]
DURUMLAR = {
    200: 70,
    301: 5,
    304: 5,
    400: 3,
    401: 2,
    403: 2,
    404: 6,
    500: 4,
    502: 2,
    503: 1,
}
YANIT_SURESI = [(5, 50)] * 85 + [(100, 500)] * 10 + [(600, 2000)] * 5

def agirlikli_secim(agirlik_sozlugu):
    secimler = []
    for deger, agirlik in agirlik_sozlugu.items():
        secimler.extend([deger] * agirlik)
    return random.choice(secimler)

if __name__ == "__main__":
    print("[uretici] Log uretimine baslandi.")
    while True:
        zaman = (datetime.now() + timedelta(seconds=random.randint(-2, 0))).strftime("%Y-%m-%d %H:%M:%S")
        ip = random.choice(IPLER)
        metod = random.choice(METOTLAR)
        url = random.choice(URL_LER)
        durum = agirlikli_secim(DURUMLAR)
        alt, ust = random.choice(YANIT_SURESI)
        ms = random.randint(alt, ust)
        print(f"[{zaman}] {metod} {url} {durum} {ms}ms", flush=True)
        time.sleep(random.uniform(0.2, 0.8))
