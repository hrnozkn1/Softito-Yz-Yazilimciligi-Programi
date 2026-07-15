# 01 - Belge Tabanlı Sorgulama

OpenAI API ile çalışan giriş seviyesi RAG. FAISS ile vektör araması, GPT ile yanıt üretimi.

## Kurulum

```bash
cd rag/01-belge-tabanli-sorgulama
pip install -r requirements.txt
```

`.env` dosyasına API anahtarını ekle:
```
OPENAI_API_KEY=sk-...
```

## Çalıştırma

```bash
python belge_tabanli_sorgulama.py
```

Script örnek belgeleri otomatik oluşturur, index'ler ve interaktif soru-cevap moduna geçer. Kaynak belgeleri ve benzerlik skorlarını gösterir.

## Dosyalar

```
├── belge_tabanli_sorgulama.py   # Ana script
├── requirements.txt
├── belgeler/                    # TXT belgeler (otomatik oluşur)
└── .gitignore
```

## Kullanılanlar

`LangChain` · `OpenAI API` · `FAISS`
