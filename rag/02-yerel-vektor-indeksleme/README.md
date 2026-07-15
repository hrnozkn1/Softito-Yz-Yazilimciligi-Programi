# 02 - Yerel Vektör İndeksleme

Tamamen yerel RAG. HuggingFace embedding + ChromaDB (diskte kalıcı) + Gradio web arayüzü. PDF/TXT desteği.

## Kurulum

```bash
cd rag/02-yerel-vektor-indeksleme
pip install -r requirements.txt
```

## Çalıştırma

```bash
python yerel_vektor_indeksleme.py
```

İlk çalıştırmada embedding modeli indirilir (~80 MB) ve index oluşturulur. Sonrakilerde doğrudan arayüz açılır. Tarayıcıda `http://127.0.0.1:7860`

## Dosyalar

```
├── yerel_vektor_indeksleme.py   # Ana script
├── requirements.txt
├── belgeler/                    # PDF ve TXT belgeler
├── chroma_db/                   # ChromaDB index (gitignored)
└── .gitignore
```

## Kullanılanlar

`LangChain` · `HuggingFace` · `ChromaDB` · `Gradio`
