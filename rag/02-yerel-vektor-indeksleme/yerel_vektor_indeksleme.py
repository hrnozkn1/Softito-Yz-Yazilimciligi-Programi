"""
Yerel Vektor Indeksleme — Tamamen Yerel RAG
=============================================
HuggingFace embedding + ChromaDB + Gradio arayuz.
PDF ve TXT destegi. Diske kalici index. API gerekmez.

Ilk calistirmada:
  1. HuggingFace embedding modeli indirilir (~80 MB)
  2. Ornek belgeler olusturulur
  3. Belgeler vektorlestirilip ChromaDB'ye kaydedilir

Sonraki calistirmalarda mevcut index yuklenir.
"""

import os
from pathlib import Path

BELGELER_KLASORU = "belgeler"
CHROMA_DIZINI = "chroma_db"

ORNEK_BELGELER = {
    "yapay_zeka_tarihi.txt": """
Yapay Zeka Tarihi

Yapay zeka (AI) kavrami ilk kez 1956'da Dartmouth Konferansi'nda John McCarthy
tarafindan ortaya atildi. Ilk donemde sembolik AI ve kural tabanli sistemler
one cikti.

1980'lerde uzman sistemler (expert systems) populer oldu. MYCIN gibi sistemler
tip alaninda basarili sonuclar verdi. Ancak kural tabanli sistemlerin olceklenme
sorunu nedeniyle 1990'larda ikinci bir "AI kisi" yasandi.

2012'de AlexNet'in ImageNet yarismasini kazanmasiyla derin ogrenme cagi basladi.
GPU'larin gelismesi ve buyuk veri setlerinin ulasilabilir olmasi, sinir aglarinin
yeniden dogusunu sagladi.

2017'de Google'in "Attention Is All You Need" makalesi Transformer mimarisini
tanitti. Bu, BERT, GPT ve tum modern dil modellerinin temelini olusturdu.

2022'de ChatGPT'nin piyasaya surulmesiyle yapay zeka genel halk tarafindan
benimsendi. Artik AI, kod yazmaktan icerik uretmeye kadar pek cok alanda
aktif olarak kullaniliyor.
""",

    "transformer_nedir.txt": """
Transformer Mimarisi

Transformer, 2017'de Google Brain ekibi tarafindan gelistirilen bir sinir agi
mimarisidir. RNN ve LSTM'lerin aksine, diziyi sirayla degil paralel olarak
isler. Bu sayede egitim suresi dramatik sekilde azalir.

Transformer'in temel bilesenleri:
- Self-Attention: Her token'in diger tum token'larla iliskisini hesaplar
- Multi-Head Attention: Farkli iliski tiplerini paralel olarak ogrenir
- Positional Encoding: Sira bilgisini modele ekler (Transformer'in dogal sira algisi yoktur)
- Feed-Forward Network: Her pozisyonu bagimsiz olarak isler
- Layer Normalization: Egitimi stabilize eder

Encoder-decoder yapisindadir:
- Encoder: Giris metnini isler, baglamsal temsiller uretir
- Decoder: Encoder ciktisini kullanarak yeni token'lar uretir

BERT sadece encoder, GPT sadece decoder kullanir. T5 ve BART ise tam encoder-decoder
mimarisini kullanir.
""",

    "chromadb.txt": """
ChromaDB — Acik Kaynak Vektor Veritabani

ChromaDB, yapay zeka uygulamalari icin ozel olarak gelistirilmis acik kaynakli
bir vektor veritabanidir. LangChain ve LlamaIndex ile dogal entegrasyon sunar.

Temel ozellikler:
- Diske kalici (persistent) depolama: Index sunucu kapatilsa bile korunur
- Metadata filtreleme: Belgeleri kategori, tarih, kaynak gibi alanlara gore filtreleyebilirsiniz
- Embedding API'si: Kendi embedding modelinizi getirebilirsiniz
- Python ve JavaScript SDK: Her iki ekosistemde de kullanilabilir

ChromaDB ozellikle RAG uygulamalari icin idealdir. FAISS'ten farki, verileri
diskte saklamasi ve daha yuksek seviyeli bir API sunmasidir.

Basit bir kullanim:
```python
import chromadb
client = chromadb.PersistentClient(path="./db")
collection = client.create_collection("belgeler")
collection.add(documents=["..."], ids=["1"])
sonuclar = collection.query(query_texts=["..."], n_results=3)
```
""",

    "gradio.txt": """
Gradio — Makine Ogrenmesi Icin Web Arayuzu

Gradio, ML modelleri icin hizla web arayuzu olusturmayi saglayan bir Python
kutuphanesidir. HuggingFace tarafindan gelistirilmektedir.

Temel ozellikler:
- Birkac satir kodla interaktif arayuz olusturma
- Text, image, audio, video gibi bircok giris/cikis turu destegi
- HuggingFace Spaces'te tek tikla deploy etme
- API endpoint'i otomatik olusturma
- Streaming yanit destegi (LLM'ler icin onemli)

Ornek kullanim:
```python
import gradio as gr

def selam_ver(isim):
    return f"Merhaba {isim}!"

gr.Interface(fn=selam_ver, inputs="text", outputs="text").launch()
```

Gradio, prototip asamasindaki ML projelerini hizla demo haline getirmek icin
en populer aractir. Alternatifleri Streamlit ve Dash'tir.
"""
}


def ornek_belgeleri_olustur():
    klasor = Path(BELGELER_KLASORU)
    klasor.mkdir(exist_ok=True)
    for dosya_adi, icerik in ORNEK_BELGELER.items():
        yol = klasor / dosya_adi
        if not yol.exists():
            yol.write_text(icerik.strip(), encoding="utf-8")
            print(f"  Olusturuldu: {dosya_adi}")


def belgeleri_yukle(klasor):
    from langchain_community.document_loaders import TextLoader, PyPDFLoader

    belgeler = []
    for dosya in Path(klasor).glob("*"):
        if dosya.suffix.lower() == ".txt":
            loader = TextLoader(str(dosya), encoding="utf-8")
        elif dosya.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(dosya))
        else:
            continue
        belgeler.extend(loader.load())
        print(f"  Yuklendi: {dosya.name} ({len(belgeler)} belge)")

    return belgeler


def index_olustur_veya_yukle(embedding_modeli, chunk_size=500, chunk_overlap=100):
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if Path(CHROMA_DIZINI).exists() and list(Path(CHROMA_DIZINI).iterdir()):
        print("\n[ok] Mevcut ChromaDB index bulundu, yukleniyor...")
        return Chroma(
            persist_directory=CHROMA_DIZINI,
            embedding_function=embedding_modeli,
        )

    print("\n[*] Index bulunamadi, sifirdan olusturuluyor...")
    ornek_belgeleri_olustur()

    print("\n[1] Belgeler yukleniyor...")
    belgeler = belgeleri_yukle(BELGELER_KLASORU)
    if not belgeler:
        print("  HATA: Hic belge bulunamadi!")
        return None
    print(f"  Toplam {len(belgeler)} belge yuklendi.")

    print(f"\n[2] Belgeler parcalaniyor (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    parcalar = splitter.split_documents(belgeler)
    print(f"  {len(parcalar)} parca olusturuldu.")

    print("\n[3] Embedding olusturuluyor ve ChromaDB'ye kaydediliyor...")
    vektor_db = Chroma.from_documents(
        parcalar, embedding_modeli,
        persist_directory=CHROMA_DIZINI,
    )
    print(f"  ChromaDB index kaydedildi: {CHROMA_DIZINI}/")
    return vektor_db


def llm_yukle():
    model_adi = "google/flan-t5-base"

    try:
        from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
        from langchain_huggingface import HuggingFacePipeline

        print(f"\n[*] HuggingFace modeli yukleniyor: {model_adi}")
        print("    (ilk calistirmada ~1 GB indirilir, sonrakilerde hazirdir)")

        tokenizer = AutoTokenizer.from_pretrained(model_adi)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_adi)

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.1,
        )
        return HuggingFacePipeline(pipeline=pipe)

    except ImportError:
        print("\n[!] transformers/torch yuklu degil.")
        print("    Yerel LLM icin: pip install transformers torch accelerate")
        raise
    except Exception as e:
        print(f"\n[!] HuggingFace model yuklenemedi: {e}")
        print("    OpenAI fallback deneniyor...")
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
        raise RuntimeError("LLM yuklenemedi. Transformers veya OpenAI API anahtari gerekli.")


def gradio_ui_baslat(vektor_db, llm):
    import gradio as gr
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = PromptTemplate.from_template("""Asagidaki baglam bilgilerini kullanarak soruyu Turkce yanitla.
Baglamda yeterli bilgi yoksa 'Bu bilgi baglamda yer almiyor.' diye belirt.

Baglam:
{context}

Soru: {question}

Yanit:""")

    parser = StrOutputParser()

    def soru_yanitla(soru, gecmis):
        if not soru.strip():
            return "Lutfen bir soru yazin."

        benzer_parcalar = vektor_db.similarity_search_with_score(soru, k=3)

        kaynak_bilgisi = []
        kaynak_metinleri = []
        for i, (doc, score) in enumerate(benzer_parcalar, 1):
            kaynak = Path(doc.metadata.get("source", "?")).name
            kaynak_bilgisi.append(f"[{i}] {kaynak}  (benzerlik: {score:.4f})")
            kaynak_metinleri.append(doc.page_content)

        baglam = "\n\n---\n\n".join(kaynak_metinleri)

        try:
            yanit = (prompt | llm | parser).invoke({
                "context": baglam,
                "question": soru,
            })
        except Exception as e:
            yanit = f"Yanit uretilemedi: {e}"

        kaynak_bolumu = "\n".join(kaynak_bilgisi)
        tam_yanit = f"{yanit}\n\n---\nKaynak belgeler:\n{kaynak_bolumu}"

        return tam_yanit

    iface = gr.ChatInterface(
        fn=soru_yanitla,
        title="Yerel RAG — Belge Sorgulama",
        description="Belgeleriniz uzerinde soru-cevap yapin. Tamamen yerel calisir.",
        theme="soft",
        examples=[
            "Transformer mimarisi nedir?",
            "ChromaDB ne ise yarar?",
            "Yapay zeka tarihi hakkinda bilgi ver.",
        ],
    )

    print("\n" + "=" * 50)
    print("  Gradio arayuzu baslatiliyor...")
    print("  Tarayicida acin: http://127.0.0.1:7860")
    print("=" * 50)

    iface.launch(share=False)


if __name__ == "__main__":
    print("=" * 55)
    print("  RAG — Yerel Vektor Indeksleme")
    print("=" * 55)

    # 1. HuggingFace embedding modeli
    print("\n[1] HuggingFace embedding modeli yukleniyor (all-MiniLM-L6-v2)...")
    print("    Ilk calistirmada ~80 MB indirilir.")
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_modeli = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    print("    Embedding modeli hazir.")

    # 2. ChromaDB index (olustur veya yukle)
    vektor_db = index_olustur_veya_yukle(embedding_modeli)
    if vektor_db is None:
        raise SystemExit(1)

    # 3. LLM yukle (once HuggingFace, yoksa OpenAI)
    llm = llm_yukle()
    print("    LLM hazir.")

    # 4. Gradio arayuzunu baslat
    gradio_ui_baslat(vektor_db, llm)
