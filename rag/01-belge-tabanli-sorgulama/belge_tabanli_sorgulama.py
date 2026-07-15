"""RAG: belgeler → chunk → embedding → FAISS → arama → LLM → yanıt"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

BELGELER_KLASORU = "belgeler"

ORNEK_BELGELER = {
    "vektor_db.txt": """
Vektör Veritabanları

Vektör veritabanı, metin, görüntü veya ses gibi yapılandırılmamış verileri
sayısal vektörlere dönüştürerek saklayan özel bir veritabanı türüdür.

FAISS (Facebook AI Similarity Search), Meta tarafından geliştirilen açık
kaynaklı bir vektör arama kütüphanesidir. Milyarlarca vektör arasında
milisaniyeler içinde en yakın komşu araması yapabilir.

ChromaDB, vektörleri diskte saklayan ve metadata filtreleme desteği sunan
bir vektör veritabanıdır. FAISS'e göre daha yüksek seviyeli bir API sunar.
""",

    "rag_nedir.txt": """
RAG (Retrieval-Augmented Generation)

RAG, büyük dil modellerinin (LLM) harici bilgi kaynaklarından bilgi alarak
yanıt üretmesini sağlayan bir tekniktir. 2020'de Meta AI tarafından tanıtılmıştır.

RAG olmadan LLM'ler sadece eğitim verilerindeki bilgiyle yanıt verir.
Bu, güncel olmayan veya yanlış bilgilere (halüsinasyon) yol açabilir.

RAG'ın temel avantajları:
- Güncel bilgi: Belgeler anlık güncellenebilir
- Kaynak gösterilebilirlik: Yanıtın nereden geldiği bellidir
- Daha az halüsinasyon: Model bağlama dayalı yanıt verir
- Domain-specific: Kendi belgelerinle özelleştirilebilir
""",

    "chunking.txt": """
Metin Parçalama (Chunking)

Uzun belgeleri LLM'in context penceresine sığacak küçük parçalara bölme
işlemine chunking denir.

RecursiveCharacterTextSplitter en yaygın kullanılan yöntemdir.
Sırasıyla paragraf, cümle ve kelime bazında bölme yapar.
Bu sayede anlam bütünlüğü en iyi şekilde korunur.

Örtüşme (overlap): Ardışık parçalar arasında belirli miktarda metin
tekrarı bırakılır. Bu, bağlamın parçalar arasında kopmasını önler.
""",

    "embedding.txt": """
Metin Gömme (Embedding)

Embedding, metni sayısal bir vektöre dönüştürme işlemidir.
Bu vektörler metnin anlamsal özelliklerini taşır.

Popüler embedding modelleri:
- OpenAI text-embedding-ada-002: 1536 boyut, ücretli, yüksek kalite
- all-MiniLM-L6-v2: 384 boyut, ücretsiz, hafif ve hızlı
- Cohere Embed v3: 1024 boyut, çok dilli desteği güçlü
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
    belgeler = []
    for dosya in Path(klasor).glob("*.txt"):
        belgeler.extend(TextLoader(str(dosya), encoding="utf-8").load())
    return belgeler


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY gerekli (.env dosyasına ekle)")

    print("RAG — Belge Tabanlı Sorgulama\n")

    ornek_belgeleri_olustur()
    belgeler = belgeleri_yukle(BELGELER_KLASORU)
    print(f"Belgeler: {len(belgeler)} adet yüklendi")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    parcalar = splitter.split_documents(belgeler)
    print(f"Parçalar: {len(parcalar)} adet oluşturuldu")

    embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-ada-002")
    vektor_db = FAISS.from_documents(parcalar, embeddings)
    print("FAISS index hazır")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)

    prompt = PromptTemplate.from_template("""Aşağıdaki bağlamı kullanarak soruyu Türkçe yanıtla.
Bağlamda bilgi yoksa belirt.

Bağlam:
{context}

Soru: {question}

Yanıt:""")

    parser = StrOutputParser()

    print("\nHazır. Soru sor (çıkmak için q)\n")

    while True:
        soru = input("Soru: ").strip()
        if not soru:
            continue
        if soru.lower() in ("q", "quit", "çıkış"):
            break

        sonuclar = vektor_db.similarity_search_with_score(soru, k=3)
        kaynak_metinleri = []

        print(f"\n{len(sonuclar)} kaynak bulundu:")
        for i, (doc, score) in enumerate(sonuclar, 1):
            kaynak = Path(doc.metadata.get("source", "?")).name
            print(f"  [{i}] {kaynak} (benzerlik: {score:.4f})")
            kaynak_metinleri.append(doc.page_content)

        baglam = "\n\n---\n\n".join(kaynak_metinleri)
        yanit = (prompt | llm | parser).invoke({"context": baglam, "question": soru})
        print(f"\nYanıt: {yanit}\n")
