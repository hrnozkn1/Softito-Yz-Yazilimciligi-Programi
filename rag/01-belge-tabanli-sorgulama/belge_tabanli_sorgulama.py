"""
Belge Tabanlı Sorgulama — RAG (Retrieval-Augmented Generation)
===============================================================
En sade haliyle RAG pipeline'ı:
  belgeler → chunk → embedding → FAISS → soru → arama → prompt → LLM → yanıt

Her adım konsola yazdırılır, kaynak belgeler gösterilir.
"""

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
from langchain_core.runnables import RunnablePassthrough

BELGELER_KLASORU = "belgeler"

ORNEK_BELGELER = {
    "vektor_db.txt": """
Vektör Veritabanları

Vektör veritabanı, metin, görüntü veya ses gibi yapılandırılmamış verileri
sayısal vektörlere dönüştürerek saklayan özel bir veritabanı türüdür.

Bir embedding modeli (örneğin text-embedding-ada-002) metni 1536 boyutlu
bir vektöre çevirir. Bu vektörler, anlamsal olarak benzer metinlerin
birbirine yakın konumlanacağı şekilde eğitilmiştir.

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

RAG, fine-tuning'e göre çok daha ucuz ve hızlıdır. Modeli yeniden eğitmek
yerine sadece belgeleri güncellersiniz.
""",

    "chunking.txt": """
Metin Parçalama (Chunking)

Uzun belgeleri LLM'in context penceresine sığacak küçük parçalara bölme
işlemine chunking denir.

İyi bir chunking stratejisi RAG performansını doğrudan etkiler:
- Çok küçük chunk'lar: Bağlam kaybı, anlamsız parçalar
- Çok büyük chunk'lar: İlgisiz bilgi karışır, token israfı

RecursiveCharacterTextSplitter en yaygın kullanılan yöntemdir.
Sırasıyla paragraf, cümle ve kelime bazında bölme yapar.
Bu sayede anlam bütünlüğü en iyi şekilde korunur.

Örtüşme (overlap): Ardışık parçalar arasında belirli miktarda metin
tekrarı bırakılır. Bu, bağlamın parçalar arasında kopmasını önler.
Genelde chunk boyutunun %10-20'si kadar örtüşme önerilir.
""",

    "embedding.txt": """
Metin Gömme (Embedding)

Embedding, metni sayısal bir vektöre dönüştürme işlemidir.
Bu vektörler metnin anlamsal özelliklerini taşır.

Örnek:
- "kedi" ve "köpek" embedding'leri birbirine yakındır (ikisi de evcil hayvan)
- "kedi" ve "otomobil" embedding'leri birbirine uzaktır

Popüler embedding modelleri:
- OpenAI text-embedding-ada-002: 1536 boyut, ücretli, yüksek kalite
- all-MiniLM-L6-v2: 384 boyut, ücretsiz, hafif ve hızlı
- Cohere Embed v3: 1024 boyut, çok dilli desteği güçlü

Embedding kalitesi, vektör aramasının başarısını belirleyen en kritik faktördür.
""",
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
    belgeler = []
    for dosya in Path(klasor).glob("*.txt"):
        loader = TextLoader(str(dosya), encoding="utf-8")
        belgeler.extend(loader.load())
        print(f"  Yuklendi: {dosya.name}")
    return belgeler


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("HATA: OPENAI_API_KEY ortam degiskeni veya .env dosyasi gerekli.")
        raise SystemExit(1)

    print("=" * 55)
    print("  RAG — Belge Tabanlı Sorgulama")
    print("=" * 55)

    # 1. Örnek belgeleri oluştur
    print("\n[1] Örnek belgeler olusturuluyor...")
    ornek_belgeleri_olustur()

    # 2. Belgeleri yükle
    print("\n[2] Belgeler yukleniyor...")
    belgeler = belgeleri_yukle(BELGELER_KLASORU)
    print(f"  Toplam {len(belgeler)} belge yuklendi.")

    # 3. Parçalara böl
    print("\n[3] Belgeler parcalaniyor (chunk_size=500, overlap=100)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )
    parcalar = splitter.split_documents(belgeler)
    print(f"  {len(parcalar)} parca olusturuldu.")
    for i, p in enumerate(parcalar):
        print(f"    Parca {i+1}: {p.page_content[:60]}... [{p.metadata.get('source', '?')}]")

    # 4. Embedding + FAISS
    print("\n[4] Embedding olusturuluyor (text-embedding-ada-002)...")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-ada-002")
    vektor_db = FAISS.from_documents(parcalar, embeddings)
    print(f"  {len(parcalar)} vektor FAISS'e kaydedildi.")

    # 5. LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)

    # 6. Prompt
    prompt = PromptTemplate.from_template("""Asagidaki baglam bilgilerini kullanarak soruyu Turkce yanitla.
Baglamda yeterli bilgi yoksa 'Bu bilgi baglamda yer almiyor.' yazarak belirt.

Baglam:
{context}

Soru: {question}

Yanit:""")

    # 7. Çıktı ayrıştırıcı
    parser = StrOutputParser()

    print("\n" + "=" * 55)
    print("  Sistem hazir. Soru sorabilirsiniz. (cikmak icin 'q')")
    print("=" * 55)

    while True:
        soru = input("\n Soru: ").strip()
        if not soru:
            continue
        if soru.lower() in ("q", "quit", "cikis", "exit"):
            break

        # 8. Retrieval: soruya en benzer 3 parçayı bul
        sonuclar = vektor_db.similarity_search_with_score(soru, k=3)
        print(f"\n  {len(sonuclar)} kaynak bulundu:")

        kaynak_metinleri = []
        for i, (doc, score) in enumerate(sonuclar, 1):
            kaynak = doc.metadata.get("source", "?").replace("\\", "/")
            print(f"    [{i}] {kaynak}  (benzerlik: {score:.4f})")
            kaynak_metinleri.append(doc.page_content)

        # 9. Generasyon: bağlam + soru → LLM
        baglam = "\n\n---\n\n".join(kaynak_metinleri)
        yanit = (prompt | llm | parser).invoke({
            "context": baglam,
            "question": soru,
        })

        print(f"\n  Yanit: {yanit}")

    print("\nProgram sonlandi.")
