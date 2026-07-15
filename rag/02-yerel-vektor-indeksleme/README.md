# 02 - Yerel Vektör İndeksleme

> Tamamen yerel çalışan RAG sistemi. HuggingFace embedding + ChromaDB + Gradio arayüz. İnternet yoksa bile çalışır.

---

## Ne Yapıyor?

API anahtarı gerektirmeyen, tamamen yerel bir RAG sistemi.

Adım adım:
1. `belgeler/` klasöründeki PDF ve TXT dosyalarını yükle
2. `RecursiveCharacterTextSplitter` ile parçalara böl
3. HuggingFace `all-MiniLM-L6-v2` ile embedding oluştur (384 boyut, ~80 MB)
4. ChromaDB'ye kaydet (disk üzerinde kalıcı index)
5. Sonraki çalıştırmalarda mevcut index'i yükle (embedding tekrar üretilmez)
6. Gradio ile tarayıcıda soru-cevap arayüzü
7. Her yanıtta kaynak belgeleri ve benzerlik skorunu göster

**Önemli:** İlk çalıştırmada embedding modeli indirilir (~80 MB). Sonraki çalıştırmalarda hazırdır.

---

## Kurulum

```bash
cd rag/02-yerel-vektor-indeksleme
pip install -r requirements.txt
```

---

## Çalıştırma

```bash
python yerel_vektor_indeksleme.py
```

İlk çalıştırmada:
1. HuggingFace modeli indirilir (`all-MiniLM-L6-v2`)
2. Örnek belgeler oluşturulur (`belgeler/`)
3. Belgeler vektörleştirilip ChromaDB'ye kaydedilir

Tarayıcıda `http://127.0.0.1:7860` adresinde Gradio arayüzü açılır.

Sonraki çalıştırmalarda index zaten hazırdır, doğrudan arayüz açılır.

---

## Dosya Yapısı

```
02-yerel-vektor-indeksleme/
├── README.md
├── requirements.txt
├── yerel_vektor_indeksleme.py
├── .gitignore
├── belgeler/              # TXT + PDF belgelerin
└── chroma_db/             # ChromaDB index (otomatik oluşur, gitignored)
```

---

## Kullanılanlar

`Python 3.11` · `LangChain` · `HuggingFace Embeddings` · `ChromaDB` · `Gradio`
