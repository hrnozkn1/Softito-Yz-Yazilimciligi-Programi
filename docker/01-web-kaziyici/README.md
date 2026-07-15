# 01 - Web Kazıyıcı

> 3 haber sitesinden başlık çeker, en sık kelimeleri sayar, kelime bulutu çıkarır. Docker ile çalışır.

---

## Ne Yapıyor?

- 3 Türkçe haber sitesinden manşetleri çeker (`requests` + `BeautifulSoup`)
- Metinleri temizler, stop-word'leri atar
- En çok geçen 30 kelimeyi CSV olarak kaydeder
- Basit bir kelime sıklığı grafiği çizer

**Önemli:** Bu bir NLP projesi değil. `split()`, `lower()`, `replace()` gibi düz string işlemleri kullanır.

---

## Dockerfile Ne Yapıyor?

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
CMD ["python", "main.py"]
```

Adım adım:
1. Hafif Python imajı
2. Bağımlılıkları kur
3. Scripti kopyala
4. Çalıştır

Volume ile `cikti/` klasörünü dışarı bağlamazsan sonuçlar konteyner içinde kalır.

---

## Çalıştırma

```bash
cd 01-web-kaziyici
docker build -t web-kaziyici .
docker run --rm -v "$(pwd)/cikti:/app/cikti" web-kaziyici
```

Çıktılar `cikti/` klasöründe:
- `kelime_sikliklari.csv` — En sık 30 kelime
- `kelime_grafigi.png` — Kelime sıklığı bar chart
- `tum_basliklar.txt` — Ham başlıklar

---

## Dosya Yapısı

```
01-web-kaziyici/
├── Dockerfile
├── main.py
├── requirements.txt
├── .gitignore
├── README.md
└── cikti/                  # Volume mount noktası
```

---

## Kullanılanlar

`Python 3.11` · `Docker` · `BeautifulSoup4` · `requests` · `matplotlib`
