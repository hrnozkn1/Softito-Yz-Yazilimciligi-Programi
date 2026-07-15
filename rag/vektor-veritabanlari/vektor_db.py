"""Vektör Veritabanları: ChromaDB ile embedding ve benzerlik araması"""
import os
import numpy as np
from pathlib import Path

os.makedirs("cikti", exist_ok=True)

BELGELER = [
    "İstanbul, Türkiye'nin en kalabalık şehri ve ekonomik merkezidir.",
    "Ankara, Türkiye'nin başkenti ve ikinci büyük şehridir.",
    "Python, yapay zeka ve veri bilimi için en popüler programlama dilidir.",
    "PyTorch ve TensorFlow, derin öğrenme için kullanılan framework'lerdir.",
    "Docker, uygulamaları konteyner içinde izole çalıştırmayı sağlar.",
    "Konteyner teknolojisi, mikroservis mimarisinin temelini oluşturur.",
    "FAISS, Facebook tarafından geliştirilen vektör arama kütüphanesidir.",
    "ChromaDB, açık kaynaklı bir vektör veritabanıdır.",
    "RAG, belgelerden bilgi çekip LLM'e bağlam olarak veren tekniktir.",
    "Semantik arama, kelime değil anlam benzerliğine dayalı arama yapar.",
    "Boğaziçi Köprüsü, İstanbul'un Avrupa ve Asya yakalarını birleştirir.",
    "Express.js, Node.js için web uygulama framework'üdür.",
]

def kosinus_benzerligi(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

def demo():
    from sentence_transformers import SentenceTransformer

    print("Embedding modeli yükleniyor...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(BELGELER)
    print(f"  {len(embeddings)} belge vektörlendi (boyut: {embeddings.shape[1]})")

    sorular = [
        "Türkiye'nin başkenti neresi?",
        "Hangi programlama dili AI için popüler?",
        "Vektör araması nasıl yapılır?",
    ]

    for soru in sorular:
        soru_vec = model.encode([soru])[0]
        benzerlikler = [(i, kosinus_benzerligi(soru_vec, emb)) for i, emb in enumerate(embeddings)]
        benzerlikler.sort(key=lambda x: x[1], reverse=True)
        print(f"\nSoru: '{soru}'")
        for rank, (idx, score) in enumerate(benzerlikler[:3]):
            print(f"  [{rank+1}] skor={score:.4f} | {BELGELER[idx]}")

def chromadb_demo():
    print("\n\nChromaDB Demo")
    print("=" * 40)
    import chromadb
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path="chroma_db")
    try:
        client.delete_collection("belgeler")
    except: pass
    collection = client.create_collection("belgeler")

    embeddings = model.encode(BELGELER).tolist()
    collection.add(
        documents=BELGELER,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(BELGELER))],
        metadatas=[{"kategori": "turkiye" if "Türkiye" in b or "İstanbul" in b or "Ankara" in b else "teknoloji"} for b in BELGELER],
    )

    soru = "İstanbul hakkında bilgi ver"
    soru_vec = model.encode([soru]).tolist()
    sonuclar = collection.query(query_embeddings=soru_vec, n_results=3)

    print(f"\nSoru: '{soru}'")
    for i, (doc, score) in enumerate(zip(sonuclar["documents"][0], sonuclar["distances"][0])):
        print(f"  [{i+1}] skor={score:.4f} | {doc}")

    # Metadata filtreleme
    print("\nFiltreleme: sadece 'turkiye' kategorisi")
    filtered = collection.query(query_embeddings=soru_vec, n_results=3, where={"kategori": "turkiye"})
    for i, doc in enumerate(filtered["documents"][0]):
        print(f"  [{i+1}] {doc}")

    print("\nChromaDB index kaydedildi: chroma_db/")

if __name__ == "__main__":
    print("Vektör Veritabanları\n")
    demo()
    chromadb_demo()
