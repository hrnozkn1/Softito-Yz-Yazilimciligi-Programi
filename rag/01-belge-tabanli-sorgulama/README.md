# 01 - Belge Tabanlı Sorgulama

> RAG'ın temellerini kavramak için en sade hali. OpenAI API ile embedding + LLM, FAISS ile vektör araması.

---

## Ne Yapıyor?

Kendi oluşturacağın bilgi metinlerini (TX) okuyup soru-cevap yapmanı sağlayan bir RAG sistemi.

Adım adım:
1. `belgeler/` klasöründeki TXT dosyalarını yükle
2. `RecursiveCharacterTextSplitter` ile 500 karakterlik parçalara böl
3. OpenAI `text-embedding-ada-002` ile her parçayı vektöre çevir
4. FAISS'te index'le
5. Soruyu vektöre çevir → en benzer 3 parçayı bul
6. Parçalar + soru → prompt → GPT'ye gönder → yanıt
7. Kaynak parçaları göster (şeffaflık)

**Amaç:** RAG pipeline'ının her adımını anlamak. Production değil, eğitim amaçlıdır.

---

## Kurulum

```bash
cd rag/01-belge-tabanli-sorgulama
pip install -r requirements.txt
```

`.env` dosyası oluştur veya ortam değişkeni olarak ayarla:
```
OPENAI_API_KEY=sk-...
```

---

## Çalıştırma

```bash
python belge_tabanli_sorgulama.py
```

Program:
1. Örnek belgeleri oluşturur (`belgeler/` klasörüne)
2. Belgeleri parçalar, vektörleştirir
3. İnteraktif soru-cevap moduna geçer

Örnek kullanım:
```
💬 Soru: Vektör veritabanı nedir?
📚 3 kaynak bulundu (benzerlik skoru: 0.89-0.94)

🤖 Yanıt:
Vektör veritabanı, yapılandırılmamış verileri...
[Kaynak: belgeler/vektor_db.txt]
```

Çıkmak için `q` yaz.

---

## Dosya Yapısı

```
01-belge-tabanli-sorgulama/
├── README.md
├── requirements.txt
├── belge_tabanli_sorgulama.py
├── .gitignore
└── belgeler/              # Örnek TXT'ler (script otomatik oluşturur)
```

---

## Kullanılanlar

`Python 3.11` · `LangChain` · `OpenAI API` · `FAISS`
