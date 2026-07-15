# RAG Projeleri

RAG (Retrieval-Augmented Generation), LLM'leri harici bilgi kaynaklarıyla buluşturan bir tekniktir. Belgelerden ilgili bağlamı bulup yanıta ekleyerek daha doğru sonuçlar üretir.

---

## Projeler

| # | Proje | Konu | Embedding | Vektör DB | Bağımlılık |
|---|-------|------|-----------|-----------|------------|
| 01 | [Belge Tabanlı Sorgulama](01-belge-tabanli-sorgulama) | RAG pipeline'ı adım adım kurma | OpenAI | FAISS | OpenAI API |
| 02 | [Yerel Vektör İndeksleme](02-yerel-vektor-indeksleme) | İnternetsiz, tamamen yerel RAG | HuggingFace | ChromaDB | Yok |

---

### 01 - Belge Tabanlı Sorgulama

**Dosya:** [`01-belge-tabanli-sorgulama/belge_tabanli_sorgulama.py`](01-belge-tabanli-sorgulama/belge_tabanli_sorgulama.py)

RAG'ın temellerini kavramak için en sade hali. Belge yükleme, chunking, embedding, vektör araması ve LLM ile yanıt üretme adımlarını sırayla gösterir. Her adımda ne olduğunu konsola yazdırır, kaynak belgeleri şeffaf şekilde gösterir.

**Kazanımlar:** Belge yükleme, chunking, embedding mantığı, FAISS ile vektör araması, prompt mühendisliği, kaynak gösterme.

### 02 - Yerel Vektör İndeksleme

**Dosya:** [`02-yerel-vektor-indeksleme/yerel_vektor_indeksleme.py`](02-yerel-vektor-indeksleme/yerel_vektor_indeksleme.py)

Tamamen yerel çalışan bir RAG sistemi. HuggingFace embedding + ChromaDB ile diske kalıcı index oluşturur. PDF ve TXT dosyalarını destekler. Gradio ile tarayıcıda kullanılabilir bir arayüz sunar. OpenAI API'si gerekmez.

**Kazanımlar:** Yerel embedding, kalıcı index yönetimi, PDF işleme, web arayüzü, benzerlik skoru analizi.

---

## RAG Akışı

```
Belgeler → Chunk → Embedding → Vektör DB → Kullanıcı Sorusu → Embedding → Arama → Bağlam + Soru → LLM → Yanıt
```

---

## Klasör Yapısı

```
rag/
├── README.md
├── 01-belge-tabanli-sorgulama/    # OpenAI API ile RAG (giriş)
└── 02-yerel-vektor-indeksleme/    # Tamamen yerel RAG (orta)
```

---

## Kullanılan Teknolojiler

`LangChain` · `OpenAI API` · `FAISS` · `ChromaDB` · `HuggingFace` · `Gradio` · `PyPDF`
