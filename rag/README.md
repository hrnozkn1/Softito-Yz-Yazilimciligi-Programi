# RAG Projeleri

RAG (Retrieval-Augmented Generation): LLM'leri harici belgelerle buluşturan teknik.

| # | Proje | Embedding | Vektör DB | Bağımlılık |
|---|-------|-----------|-----------|------------|
| 01 | [Belge Tabanlı Sorgulama](01-belge-tabanli-sorgulama) | OpenAI | FAISS | OpenAI API |
| 02 | [Yerel Vektör İndeksleme](02-yerel-vektor-indeksleme) | HuggingFace | ChromaDB | Yok |

### 01 - Belge Tabanlı Sorgulama

RAG pipeline'ını adım adım gösteren giriş seviyesi proje. OpenAI embedding + GPT ile çalışır.

### 02 - Yerel Vektör İndeksleme

Tamamen yerel RAG. HuggingFace embedding + ChromaDB + Gradio web arayüzü. API gerekmez.

---

**Akış:** Belgeler → Chunk → Embedding → Vektör DB → Soru → Arama → LLM → Yanıt

**Teknolojiler:** `LangChain` · `OpenAI` · `FAISS` · `ChromaDB` · `HuggingFace` · `Gradio`
