# 03 - İş Emri Kuyruğu

> 3 konteyner, 1 paylaşımlı volume. Producer üretir, consumer işler, reporter raporlar.

---

## Ne Yapıyor?

Üç Docker servisi aynı volume üzerinden konuşur:

| Servis | Görevi |
|--------|--------|
| `uretim` | Sahte sipariş JSON'ları üretir, `kuyruk/` klasörüne yazar |
| `islem` | `kuyruk/`'tan okur, her siparişi "işler" (sahte gecikme), sonucu `tamamlanan/`'a yazar |
| `rapor` | 15 saniyede bir `tamamlanan/`'ı okuyup özet dashboard basar |

**Önemli:** RabbitMQ, Redis, Kafka yok. Queue = dosya sistemi. Producer-consumer pattern'in en basit hali.

---

## docker-compose.yml Ne Yapıyor?

```yaml
services:
  uretim:
    build: .
    command: ["python", "-u", "uretim.py"]
    volumes:
      - shared_data:/app/data

  islem:
    build: .
    command: ["python", "-u", "islem.py"]
    volumes:
      - shared_data:/app/data

  rapor:
    build: .
    command: ["python", "-u", "rapor.py"]
    volumes:
      - shared_data:/app/data

volumes:
  shared_data:
```

- 3 servis de **aynı Dockerfile**'dan build edilir
- Farklı `command` ile farklı script çalıştırırlar
- `shared_data` adlı named volume hepsinde `/app/data`'ya mount edilir
- Bu sayede aynı dosyaları görüp okuyup yazabilirler

---

## Çalıştırma

```bash
cd 03-is-emri-kuyrugu
docker compose up --build
```

Çıktı örneği:

```
uretim   | [uretim] Uretim basladi. 2 saniyede bir siparis uretiyor...
islem    | [islem] Kuyruk dinleniyor...
islem    | [islem] ✓ Siparis #1 islendi (923.45 TL, 0.8s)
islem    | [islem] ✓ Siparis #2 islendi (456.12 TL, 1.1s)
rapor    | ================ RAPOR ================
rapor    | Islenen siparis : 12
rapor    | Ortalama tutar  : 742.30 TL
rapor    | Ortalama sure   : 0.92s
rapor    | =========================================
```

---

## Dosya Yapısı

```
03-is-emri-kuyrugu/
├── docker-compose.yml
├── Dockerfile
├── uretim.py
├── islem.py
├── rapor.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Kullanılanlar

`Python 3.11` · `Docker` · `Docker Compose`
