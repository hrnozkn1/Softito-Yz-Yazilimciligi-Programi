"""Yerel RAG: HuggingFace embedding + ChromaDB + Gradio. API gerekmez."""
import os
from pathlib import Path

BELGELER_KLASORU = "belgeler"
CHROMA_DIZINI = "chroma_db"

ORNEK_BELGELER = {
    "yapay_zeka_tarihi.txt": """
Yapay Zeka Tarihi

Yapay zeka (AI) kavramı ilk kez 1956'da Dartmouth Konferansı'nda John McCarthy
tarafından ortaya atıldı. İlk dönemde sembolik AI ve kural tabanlı sistemler öne çıktı.

1980'lerde uzman sistemler (expert systems) popüler oldu. 2012'de AlexNet'in
ImageNet yarışmasını kazanmasıyla derin öğrenme çağı başladı.

2017'de Transformer mimarisi tanıtıldı. 2022'de ChatGPT ile yapay zeka
genel halk tarafından benimsendi.
""",

    "transformer_nedir.txt": """
Transformer Mimarisi

Transformer, 2017'de Google Brain ekibi tarafından geliştirilen bir sinir ağı
mimarisidir. RNN ve LSTM'lerin aksine, diziyi paralel işler.

Temel bileşenleri:
- Self-Attention: Her token'in diğer token'larla ilişkisini hesaplar
- Multi-Head Attention: Farklı ilişki tiplerini paralel öğrenir
- Positional Encoding: Sıra bilgisini modele ekler

BERT sadece encoder, GPT sadece decoder kullanır.
""",

    "chromadb.txt": """
ChromaDB — Vektör Veritabanı

ChromaDB, AI uygulamaları için geliştirilmiş açık kaynaklı bir vektör
veritabanıdır. LangChain ve LlamaIndex ile doğal entegrasyon sunar.

Özellikleri: diske kalıcı depolama, metadata filtreleme, embedding API'si.
FAISS'ten farkı: verileri diskte saklar, daha yüksek seviyeli API sunar.
""",

    "gradio.txt": """
Gradio — ML Web Arayüzü

Gradio, ML modelleri için hızla web arayüzü oluşturmayı sağlayan bir Python
kütüphanesidir. HuggingFace tarafından geliştirilmektedir.

Birkaç satır kodla interaktif arayüz oluşturulabilir.
HuggingFace Spaces'te tek tıkla deploy edilebilir.
""",
}


def ornek_belgeleri_olustur():
    klasor = Path(BELGELER_KLASORU)
    klasor.mkdir(exist_ok=True)
    for ad, icerik in ORNEK_BELGELER.items():
        yol = klasor / ad
        if not yol.exists():
            yol.write_text(icerik.strip(), encoding="utf-8")


def belgeleri_yukle(klasor):
    from langchain_community.document_loaders import TextLoader, PyPDFLoader

    belgeler = []
    for dosya in Path(klasor).glob("*"):
        if dosya.suffix.lower() == ".txt":
            belgeler.extend(TextLoader(str(dosya), encoding="utf-8").load())
        elif dosya.suffix.lower() == ".pdf":
            belgeler.extend(PyPDFLoader(str(dosya)).load())
    return belgeler


def index_olustur_veya_yukle(embeddings):
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if Path(CHROMA_DIZINI).exists() and list(Path(CHROMA_DIZINI).iterdir()):
        print("Mevcut index yükleniyor...")
        return Chroma(persist_directory=CHROMA_DIZINI, embedding_function=embeddings)

    print("Index oluşturuluyor...")
    ornek_belgeleri_olustur()

    belgeler = belgeleri_yukle(BELGELER_KLASORU)
    if not belgeler:
        raise RuntimeError("Hiç belge bulunamadı")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    parcalar = splitter.split_documents(belgeler)
    print(f"  {len(parcalar)} parça, {len(belgeler)} belge")

    vektor_db = Chroma.from_documents(parcalar, embeddings, persist_directory=CHROMA_DIZINI)
    print(f"  Kaydedildi: {CHROMA_DIZINI}/")
    return vektor_db


def llm_yukle():
    try:
        from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
        from langchain_huggingface import HuggingFacePipeline

        model_adi = "google/flan-t5-base"
        print(f"LLM yükleniyor: {model_adi}...")
        tokenizer = AutoTokenizer.from_pretrained(model_adi)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_adi)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer,
                        max_new_tokens=256, temperature=0.1)
        return HuggingFacePipeline(pipeline=pipe)

    except ImportError:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            from langchain_openai import ChatOpenAI
            print("OpenAI fallback kullanılıyor...")
            return ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
        raise RuntimeError("LLM yüklenemedi. pip install transformers torch veya OPENAI_API_KEY tanımla.")


def gradio_ui_baslat(vektor_db, llm):
    import gradio as gr
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = PromptTemplate.from_template("""Aşağıdaki bağlamı kullanarak soruyu Türkçe yanıtla.
Bağlamda bilgi yoksa belirt.

Bağlam:
{context}

Soru: {question}

Yanıt:""")

    parser = StrOutputParser()

    def soru_yanitla(soru, gecmis):
        if not soru.strip():
            return "Lütfen bir soru yazın."

        sonuclar = vektor_db.similarity_search_with_score(soru, k=3)

        kaynak_bilgisi = []
        kaynak_metinleri = []
        for i, (doc, score) in enumerate(sonuclar, 1):
            kaynak = Path(doc.metadata.get("source", "?")).name
            kaynak_bilgisi.append(f"[{i}] {kaynak} (skor: {score:.4f})")
            kaynak_metinleri.append(doc.page_content)

        baglam = "\n\n---\n\n".join(kaynak_metinleri)
        try:
            yanit = (prompt | llm | parser).invoke({"context": baglam, "question": soru})
        except Exception as e:
            yanit = f"Hata: {e}"

        return f"{yanit}\n\n---\nKaynaklar:\n" + "\n".join(kaynak_bilgisi)

    iface = gr.ChatInterface(
        fn=soru_yanitla,
        title="Yerel RAG — Belge Sorgulama",
        description="Belgeleriniz üzerinde soru-cevap yapın.",
        theme="soft",
        examples=["Transformer mimarisi nedir?", "ChromaDB ne işe yarar?"],
    )
    print("\nTarayıcıda aç: http://127.0.0.1:7860\n")
    iface.launch(share=False)


if __name__ == "__main__":
    print("RAG — Yerel Vektör İndeksleme\n")

    from langchain_huggingface import HuggingFaceEmbeddings

    print("Embedding modeli yükleniyor...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    vektor_db = index_olustur_veya_yukle(embeddings)
    llm = llm_yukle()
    gradio_ui_baslat(vektor_db, llm)
