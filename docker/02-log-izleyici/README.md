# 02 - Log İzleyici

> Biri log üretir, diğeri izler. İki servis, tek Compose dosyası.

---

## Ne Yapıyor?

İki Docker servisi aynı ağda çalışır:

| Servis | Görevi |
|--------|--------|
| `log-uretici` | Her 0.5 saniyede bir sahte HTTP log satırı üretir (`stdout`'a yazar) |
| `log-izleyici` | `uretici`'nin Docker logs'unu okuyup hata pattern'lerini (404, 500, timeout) sayar |

Her 10 saniyede bir `izleyici` durum raporu basar: kaç istek geldi, kaçı hatalı, hata oranı yüzde kaç.

**Önemli:** Bu bir anomali tespiti veya ML projesi değil. Sadece string eşleştirme (`"404" in line`) ve sayaç.

---

## docker-compose.yml Ne Yapıyor?

```yaml
services:
  uretici:
    build: ./log-uretici
    container_name: uretici
  izleyici:
    build: ./log-izleyici
    container_name: izleyici
    depends_on:
      - uretici
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
```

- `izleyici` Docker socket'e erişir → `uretici` konteynerinin loglarını okuyabilir
- `depends_on` → önce `uretici` başlar, sonra `izleyici`
- Docker Compose otomatik ağ oluşturur

---

## Çalıştırma

```bash
cd 02-log-izleyici
docker compose up --build
```

Çıktı örneği:

```
uretici   | [2024-03-15 10:23:01] GET /urun/42 200 12ms
uretici   | [2024-03-15 10:23:02] POST /giris 401 5ms
izleyici  | --- RAPOR ---
izleyici  | Toplam: 145 | 200: 130 | 404: 8 | 500: 5 | 401: 2
izleyici  | Hata oranı: %10.3
izleyici  | -------------
```

Durdurmak için: `Ctrl+C`

---

## Dosya Yapısı

```
02-log-izleyici/
├── docker-compose.yml
├── log-uretici/
│   ├── Dockerfile
│   └── uretici.py
├── log-izleyici/
│   ├── Dockerfile
│   └── izleyici.py
├── .gitignore
└── README.md
```

---

## Kullanılanlar

`Python 3.11` · `Docker` · `Docker Compose`
