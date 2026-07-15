# 🐳 Docker Projeleri

Docker'ın temellerini kavramak için 3 proje. Basitten karmaşığa: tek konteyner → iki servisli → üç servisli.

---

## 📁 Projeler

| # | Proje | Konu | Docker Komutu | Seviye |
|---|-------|------|---------------|--------|
| 01 | [Web Kazıyıcı](01-web-kaziyici) | Dockerfile · Volume mount · Konteyner yaşam döngüsü | `docker build` / `docker run` | ⭐ Giriş |
| 02 | [Log İzleyici](02-log-izleyici) | Docker Compose · Servis ağı · Container logs | `docker compose up` | ⭐⭐ Orta |
| 03 | [İş Emri Kuyruğu](03-is-emri-kuyrugu) | Docker Compose · Shared volume · Producer-consumer | `docker compose up` | ⭐⭐ Orta |

---

### 01 - Web Kazıyıcı

**Dosya:** [`01-web-kaziyici/main.py`](01-web-kaziyici/main.py)

3 Türkçe haber sitesinden manşet çeker, en sık kelimeleri sayar, CSV ve grafik çıktısı verir. Tek Dockerfile ile paketlenir. Volume mount ile sonuçlar konteyner dışına alınır.

### 02 - Log İzleyici

**Dosya:** [`02-log-izleyici/docker-compose.yml`](02-log-izleyici/docker-compose.yml)

`log-uretici` sahte HTTP logları üretir, `log-izleyici` bunları okuyup hata pattern'lerini sayar. Docker Compose ile iki servis aynı ağda haberleşir.

### 03 - İş Emri Kuyruğu

**Dosya:** [`03-is-emri-kuyrugu/docker-compose.yml`](03-is-emri-kuyrugu/docker-compose.yml)

`uretim` sipariş JSON'ları üretir, `islem` bunları okuyup işler, `rapor` 15 saniyede bir özet dashboard basar. 3 servis aynı Dockerfile'dan build edilir, named volume ile aynı dosyaları görür.

---

## Çalıştırma

```bash
# Proje 1
cd docker/01-web-kaziyici
docker build -t web-kaziyici .
docker run --rm -v "$(pwd)/cikti:/app/cikti" web-kaziyici

# Proje 2
cd ../02-log-izleyici
docker compose up --build

# Proje 3
cd ../03-is-emri-kuyrugu
docker compose up --build
```

---

## Klasör Yapısı

```
docker/
├── README.md
├── 01-web-kaziyici/          # Dockerfile + volume mount (giriş)
├── 02-log-izleyici/          # Docker Compose, 2 servis (orta)
└── 03-is-emri-kuyrugu/       # Docker Compose, 3 servis + volume (orta)
```

---

## Kullanılan Araçlar

`Python 3.11` · `Docker` · `Docker Compose` · `BeautifulSoup4` · `requests` · `matplotlib` · `pandas`
