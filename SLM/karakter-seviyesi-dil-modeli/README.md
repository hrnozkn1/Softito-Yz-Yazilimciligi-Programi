# Karakter Seviyesi Dil Modeli

Wikipedia Türkçe metinleriyle karakter seviyesinde Bigram/LSTM dil modeli eğitimi. Model yeni metin üretmeyi öğrenir.

## Kapsam

- Wikipedia API ile Türkçe metin toplama
- Karakter seviyesi tokenization
- Bigram baseline model
- LSTM tabanlı dil modeli
- Eğitim ve metin üretimi

## Çalıştırma

```bash
cd SLM/karakter-seviyesi-dil-modeli
pip install -r requirements.txt
jupyter notebook
```

`karakter_slm.ipynb` dosyasını açıp hücreleri sırayla çalıştır.

## Kullanılanlar

`PyTorch` · `wikipedia-api` · `NumPy`
