#!/usr/bin/env python3
"""
word_embeddings_kapsamli.py
============================
Word Embeddings: Word2Vec, FastText ve TF-IDF ile Karşılaştırma

Word embeddings, kelimeleri anlamca yakın olanların birbirine yakın durduğu
yoğun (dense) vektörlerle temsil eden bir yöntemdir. TF-IDF'in "anlam"
eksikliğini giderir.

Veri Seti: Keloğlan Türkçe Sentiment (630K+ yorum, 3 sınıf)
           https://huggingface.co/datasets/engin1123/keloglan-turkish-sentiment-analysis-dataset

Çalıştırma:
  pip install -r requirements.txt
  python word_embeddings_kapsamli.py

Üretilenler:
  figures/ -> tüm grafikler (PNG)
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.manifold import TSNE

# ======================================================================
# AYARLAR
# ======================================================================
FIG_DIR = "figures"
RANDOM_STATE = 42
SAMPLE_SIZE = 50000       # hız için 50K örnek (toplam 630K)
EMBEDDING_DIM = 100
plt.rcParams["figure.dpi"] = 120


def baslik(metin):
    print("\n" + "=" * 70)
    print(f"  {metin}")
    print("=" * 70)


def figuru_kaydet(isim):
    os.makedirs(FIG_DIR, exist_ok=True)
    yol = os.path.join(FIG_DIR, isim)
    plt.savefig(yol, bbox_inches="tight")
    plt.close()
    print(f"  [kaydedildi] {yol}")


# ======================================================================
# 1 — TEORİ
# ======================================================================

def teori():
    """One-hot encoding ile embedding arasındaki farkı açıklar."""
    baslik("1. WORD EMBEDDINGS TEORİSİ")

    print("""
  🧩 One-Hot Encoding (TF-IDF'in temel sorunu)
     Her kelime = [0, 0, ..., 1, ..., 0]
     "kral" ve "kraliçe" arasında 0 benzerlik — anlam bilgisi yok!

  🧩 Word Embeddings
     Her kelime = [0.23, -0.45, 0.12, ..., 0.89] (100-300 boyutlu)
     Anlamca yakın kelimeler → vektör uzayında yakın
     "kral" ile "kraliçe" → yüksek cosine benzerlik

  Word2Vec (Google, 2013):
    Skip-gram: Hedef kelimeden komşu kelimeleri tahmin et
    CBOW:      Komşu kelimelerden hedef kelimeyi tahmin et

  FastText (Facebook, 2016):
    Word2Vec + karakter n-gram'ları (subword)
    "futbol" → <fu, fut, utb, tbo, bol, ol>
    Bilinmeyen kelimeleri de (OOV) tahmin edebilir
    Türkçe gibi eklemeli dillerde çok daha güçlü

  GloVe (Stanford, 2014):
    Global kelime birlikte-oluşum (co-occurrence) matrisine dayanır
    Matris faktörizasyonu + Word2Vec hibriti
  """)


# ======================================================================
# 2 — VERİ YÜKLEME
# ======================================================================

def veri_yukle():
    """Keloğlan Türkçe Sentiment veri setini yükler ve temizler."""
    baslik("2. TÜRKÇE SENTIMENT VERİ SETİ")

    etiket_map = {0: "Negatif", 1: "Nötr", 2: "Pozitif"}

    try:
        from datasets import load_dataset
        print("  HuggingFace'ten veri indiriliyor...")
        dataset = load_dataset(
            "engin1123/keloglan-turkish-sentiment-analysis-dataset",
            split="train", streaming=False
        )
        df = dataset.to_pandas()
    except Exception as e:
        print(f"  [!] Veri yüklenemedi: {e}")
        print("  Sentetik veri ile devam ediliyor...")
        np.random.seed(RANDOM_STATE)
        pozitif = ["harika ürün çok beğendim kesinlikle tavsiye ederim"] * 4000
        negatif = ["berbat bir ürün hiç beğenmedim tavsiye etmiyorum"] * 3000
        notr = ["ürün normal beklentimi karşıladı ne iyi ne kötü"] * 3000
        texts = pozitif + negatif + notr
        labels = [2]*len(pozitif) + [0]*len(negatif) + [1]*len(notr)
        df = pd.DataFrame({"text": texts, "label": labels})

    print(f"\n  Toplam örnek: {len(df):,}")
    print(f"  Sütunlar: {list(df.columns)}")

    print("\n  Etiket dağılımı:")
    for l, name in etiket_map.items():
        count = (df["label"] == l).sum()
        print(f"    {name:8s}: {count:,} (%{100*count/len(df):.1f})")

    print("\n  Örnek metinler:")
    for i in range(3):
        print(f"    [{etiket_map[df['label'].iloc[i]]}] {df['text'].iloc[i][:80]}...")

    # Hız için örnekleme
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"\n  Hız için örnekleme: {SAMPLE_SIZE:,} metin kullanılacak.")

    # Metin temizleme
    def temizle(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zöçşığüâîû0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df["clean_text"] = df["text"].apply(temizle)
    df["tokens"] = df["clean_text"].apply(lambda x: x.split())

    print("\n  Temizleme örneği:")
    print(f"    Ham:    {df['text'].iloc[0][:80]}")
    print(f"    Temiz:  {df['clean_text'].iloc[0][:80]}")

    # Etiket dağılım grafiği
    sayilar = [(df["label"] == l).sum() for l in range(3)]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(["Negatif", "Nötr", "Pozitif"], sayilar,
                   color=["#d9534f", "#f0ad4e", "#5cb85c"])
    for bar, sayi in zip(bars, sayilar):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f"{sayi:,}", ha='center', fontsize=10)
    plt.title("Sınıf Dağılımı")
    plt.tight_layout()
    figuru_kaydet("01_sinif_dagilimi.png")

    return df, etiket_map


# ======================================================================
# 3 — WORD2VEC
# ======================================================================

def word2vec_egit(df):
    """Gensim Word2Vec modelini eğitir ve benzer kelimeleri gösterir."""
    baslik("3. WORD2VEC EĞİTİMİ (Gensim)")

    try:
        from gensim.models import Word2Vec
    except ImportError:
        print("  [!] gensim yüklü değil: pip install gensim")
        return None

    sentences = df["tokens"].tolist()
    print(f"  Cümle sayısı: {len(sentences):,}")

    print("  Model eğitiliyor (Skip-gram)...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=EMBEDDING_DIM,
        window=5,
        min_count=5,
        workers=4,
        sg=1,          # Skip-gram
        seed=RANDOM_STATE,
    )

    print(f"\n  Kelime dağarcığı: {len(model.wv):,} kelime")
    print(f"  Vektör boyutu:    {model.wv.vector_size}")

    print("\n  Benzer kelime örnekleri:")
    for word in ["güzel", "kötü", "ürün", "fiyat", "harika"]:
        if word in model.wv:
            benzer = model.wv.most_similar(word, topn=5)
            print(f"    '{word}' → {', '.join([s[0] for s in benzer])}")

    return model


# ======================================================================
# 4 — FASTTEXT
# ======================================================================

def fasttext_egit(df):
    """Gensim FastText modelini eğitir ve OOV yeteneğini gösterir."""
    baslik("4. FASTTEXT EĞİTİMİ (Gensim)")

    try:
        from gensim.models import FastText
    except ImportError:
        print("  [!] gensim yüklü değil.")
        return None

    sentences = df["tokens"].tolist()
    print("  Model eğitiliyor (Skip-gram + subword)...")

    model = FastText(
        sentences=sentences,
        vector_size=EMBEDDING_DIM,
        window=5,
        min_count=5,
        workers=4,
        sg=1,
        seed=RANDOM_STATE,
    )

    print(f"\n  Kelime dağarcığı: {len(model.wv):,} kelime")
    print(f"  Vektör boyutu:    {model.wv.vector_size}")

    # OOV testi: modelin eğitimde görmediği yeni kelimeler
    print("\n  FastText'in asıl gücü: Görmediği kelimeler (OOV)")
    test_kelimeleri = ["güzellik", "kötülükçü", "ürüncükler", "harikalık"]
    for word in test_kelimeleri:
        if word in model.wv:
            benzer = model.wv.most_similar(word, topn=3)
            print(f"    '{word}' (YOK) → {', '.join([s[0] for s in benzer])}")
        else:
            print(f"    '{word}' (YOK) → vektör üretilemedi")

    return model


# ======================================================================
# 5 — t-SNE GÖRSELLEŞTİRME
# ======================================================================

def tsne_gorsellestir(model):
    """Word2Vec vektörlerini t-SNE ile 2 boyuta indirip görselleştirir."""
    baslik("5. t-SNE İLE KELİME VEKTÖRLERİNİ 2D GÖRSELLEŞTİRME")

    if model is None:
        print("  Model yok, atlanıyor.")
        return

    kelimeler = [
        "güzel", "harika", "mükemmel", "iyi", "süper",
        "kötü", "berbat", "çirkin", "pis", "rezalet",
        "fiyat", "para", "kalite", "ürün", "hizmet",
        "hızlı", "yavaş", "büyük", "küçük", "ucuz",
        "pahalı", "temiz", "kirli", "kolay", "zor",
    ]

    kelimeler = [w for w in kelimeler if w in model.wv]
    vektorler = np.array([model.wv[w] for w in kelimeler])

    print(f"  {len(kelimeler)} kelime 2 boyuta indirgeniyor...")
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=5)
    xy = tsne.fit_transform(vektorler)

    pozitif = {"güzel", "harika", "mükemmel", "iyi", "süper", "hızlı", "temiz", "kolay", "kalite"}

    plt.figure(figsize=(12, 10))
    for i, word in enumerate(kelimeler):
        renk = "#5cb85c" if word in pozitif else "#d9534f"
        plt.scatter(xy[i, 0], xy[i, 1], c=renk, s=100, alpha=0.7)
        plt.annotate(word, (xy[i, 0], xy[i, 1]), fontsize=10, ha='center',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    plt.title("Word2Vec: Türkçe Kelime Vektörleri (t-SNE)", fontsize=14)
    plt.xlabel("t-SNE Bileşen 1")
    plt.ylabel("t-SNE Bileşen 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    figuru_kaydet("02_tsne_word2vec.png")

    # Vektör aritmetiği
    print("\n  Vektör Aritmetiği (Analoji):")
    print("  'Kral - Erkek + Kadın = Kraliçe' benzeri testler...")
    analojiler = [
        ("kral", "kraliçe", "adam"),
        ("iyi", "kötü", "güzel"),
    ]
    for a, b, c in analojiler:
        if all(w in model.wv for w in [a, b, c]):
            try:
                sonuc = model.wv.most_similar(positive=[b, c], negative=[a], topn=1)
                print(f"    '{a}' - '{b}' = '{c}' - ? → '{sonuc[0][0]}' ({sonuc[0][1]:.3f})")
            except Exception:
                pass


# ======================================================================
# 6 — SINIFLANDIRMA KARŞILAŞTIRMASI
# ======================================================================

def siniflandirma_karsilastir(df, model_w2v, model_ft):
    """TF-IDF vs Word2Vec vs FastText sınıflandırma performans karşılaştırması."""
    baslik("6. TF-IDF vs WORD2VEC vs FASTTEXT SINIFLANDIRMA KARŞILAŞTIRMASI")

    X = df["clean_text"].to_numpy()
    y = df["label"].to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"  Eğitim: {len(X_tr):,} | Test: {len(X_te):,}\n")
    sonuclar = {}

    # TF-IDF + Logistic Regression
    print("  [1/3] TF-IDF + Logistic Regression...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_tr_tfidf = tfidf.fit_transform(X_tr)
    X_te_tfidf = tfidf.transform(X_te)

    clf_tfidf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf_tfidf.fit(X_tr_tfidf, y_tr)
    pred = clf_tfidf.predict(X_te_tfidf)
    sonuclar["TF-IDF"] = {
        "boyut": tfidf.max_features,
        "acc": accuracy_score(y_te, pred),
        "f1": f1_score(y_te, pred, average='weighted'),
    }
    print(f"    Doğruluk: {sonuclar['TF-IDF']['acc']:.4f}  "
          f"F1: {sonuclar['TF-IDF']['f1']:.4f}")

    # Word2Vec + Logistic Regression
    print("  [2/3] Word2Vec + Logistic Regression...")
    if model_w2v is not None:
        def dokuman_vektoru(tokens, model):
            vektorler = [model.wv[w] for w in tokens if w in model.wv]
            if not vektorler:
                return np.zeros(model.wv.vector_size)
            return np.mean(vektorler, axis=0)

        X_tr_w2v = np.array([dokuman_vektoru(t.split(), model_w2v) for t in X_tr])
        X_te_w2v = np.array([dokuman_vektoru(t.split(), model_w2v) for t in X_te])

        clf_w2v = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        clf_w2v.fit(X_tr_w2v, y_tr)
        pred = clf_w2v.predict(X_te_w2v)
        sonuclar["Word2Vec"] = {
            "boyut": EMBEDDING_DIM,
            "acc": accuracy_score(y_te, pred),
            "f1": f1_score(y_te, pred, average='weighted'),
        }
        print(f"    Doğruluk: {sonuclar['Word2Vec']['acc']:.4f}  "
              f"F1: {sonuclar['Word2Vec']['f1']:.4f}")

    # FastText + Logistic Regression
    print("  [3/3] FastText + Logistic Regression...")
    if model_ft is not None:
        def dokuman_vektoru_ft(tokens, model):
            vektorler = [model.wv[w] for w in tokens if w in model.wv]
            if not vektorler:
                return np.zeros(model.wv.vector_size)
            return np.mean(vektorler, axis=0)

        X_tr_ft = np.array([dokuman_vektoru_ft(t.split(), model_ft) for t in X_tr])
        X_te_ft = np.array([dokuman_vektoru_ft(t.split(), model_ft) for t in X_te])

        clf_ft = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        clf_ft.fit(X_tr_ft, y_tr)
        pred = clf_ft.predict(X_te_ft)
        sonuclar["FastText"] = {
            "boyut": EMBEDDING_DIM,
            "acc": accuracy_score(y_te, pred),
            "f1": f1_score(y_te, pred, average='weighted'),
        }
        print(f"    Doğruluk: {sonuclar['FastText']['acc']:.4f}  "
              f"F1: {sonuclar['FastText']['f1']:.4f}")

    # Sonuç tablosu
    print("\n  " + "=" * 60)
    print(f"  {'Yöntem':12s} | {'Boyut':8s} | {'Doğruluk':10s} | {'F1':10s}")
    print("  " + "-" * 60)
    for name, s in sonuclar.items():
        print(f"  {name:12s} | {s['boyut']:<8d} | {s['acc']:<10.4f} | {s['f1']:<10.4f}")

    # Karşılaştırma grafiği
    isimler = list(sonuclar.keys())
    accs = [sonuclar[n]["acc"] for n in isimler]
    f1s = [sonuclar[n]["f1"] for n in isimler]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(isimler))
    w = 0.35
    ax.bar(x - w/2, accs, w, label='Doğruluk', color='#0275d8')
    ax.bar(x + w/2, f1s, w, label='F1 (weighted)', color='#5cb85c')
    ax.set_xticks(x)
    ax.set_xticklabels(isimler)
    ax.set_ylabel('Skor')
    ax.set_title('TF-IDF vs Word Embeddings: Sınıflandırma Performansı')
    ax.legend()
    for i, (acc, f1) in enumerate(zip(accs, f1s)):
        ax.text(i - w/2, acc + 0.01, f"{acc:.3f}", ha='center', fontsize=9)
        ax.text(i + w/2, f1 + 0.01, f"{f1:.3f}", ha='center', fontsize=9)
    fig.tight_layout()
    figuru_kaydet("03_classification_comparison.png")

    return sonuclar


# ======================================================================
# 7 — ÖZET
# ======================================================================

def ozet(sonuclar):
    """Tüm yöntemlerin karşılaştırmalı özeti."""
    baslik("7. ÖZET VE KARŞILAŞTIRMA")

    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                            ÖZET                                     │
  ├───────────────┬──────────────────────────┬──────────────────────────┤
  │ TF-IDF        │ + Basit, hızlı           │ - Anlam bilgisi yok       │
  │               │ + Yorumlanabilir          │ - Bag-of-words           │
  │               │ + Stop-words otomatik     │ - Seyrek vektörler       │
  ├───────────────┼──────────────────────────┼──────────────────────────┤
  │ Word2Vec      │ + Anlamsal ilişkiler      │ - Eğitimi yavaş          │
  │               │ + Yoğun vektörler         │ - Daha fazla veri ister   │
  │               │ + Cosine benzerlik        │ - OOV kelimeler yok       │
  ├───────────────┼──────────────────────────┼──────────────────────────┤
  │ FastText      │ + Word2Vec + subword      │ - Daha yavaş             │
  │               │ + OOV kelimeleri bilir    │ - Daha fazla RAM          │
  │               │ + Türkçe gibi eklemeli    │                          │
  │               │   diller için ideal       │                          │
  └───────────────┴──────────────────────────┴──────────────────────────┘
  """)

    if sonuclar:
        print("  Nihai sıralama:")
        sirali = sorted(sonuclar.items(), key=lambda x: x[1]["acc"], reverse=True)
        for i, (name, s) in enumerate(sirali):
            print(f"    {i+1}. {name:12s} → Doğruluk: {s['acc']:.4f}")

    print("\n  Ne zaman hangisi?")
    print("    TF-IDF   → Küçük veri, hızlı prototip, baseline")
    print("    Word2Vec → Orta-büyük veri, anlamsal benzerlik önemliyse")
    print("    FastText → Türkçe gibi sondan eklemeli dillerde")
    print("    BERT     → En iyi performans (ama çok daha yavaş/ağır)")


# ======================================================================
# ANA PROGRAM
# ======================================================================

def main():
    print("=" * 70)
    print("    WORD EMBEDDINGS: WORD2VEC, FASTTEXT ve TF-IDF")
    print("    Kapsamlı Anlatım, Uygulama ve Karşılaştırma")
    print("=" * 70)

    teori()
    df, etiket_map = veri_yukle()
    model_w2v = word2vec_egit(df)
    model_ft = fasttext_egit(df)
    tsne_gorsellestir(model_w2v)
    sonuclar = siniflandirma_karsilastir(df, model_w2v, model_ft)
    ozet(sonuclar)

    print("\n" + "=" * 70)
    print("    TAMAMLANDI")
    print(f"    Tüm görseller '{FIG_DIR}/' klasörüne kaydedildi.")
    print("=" * 70)


if __name__ == "__main__":
    main()
