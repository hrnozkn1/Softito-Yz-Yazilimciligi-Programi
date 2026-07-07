#!/usr/bin/env python3
"""
word_embeddings_kapsamli.py
============================
Word Embeddings: Word2Vec, FastText ve TF-IDF ile Karşılaştırma

Word embeddings, kelimeleri anlamca yakın olanların birbirine yakın durduğu
yoğun (dense) vektörlerle temsil eden bir yöntemdir. TF-IDF'in "anlam"
eksikliğini giderir.

Akış:
  1) Kütüphaneler ve sabitler
  2) BÖLÜM 1 — Teori: One-hot → Embedding mantığı
  3) BÖLÜM 2 — Veri Yükleme: HuggingFace Keloğlan Türkçe Sentiment (630K)
  4) BÖLÜM 3 — Word2Vec Eğitimi (gensim)
  5) BÖLÜM 4 — FastText Eğitimi (gensim)
  6) BÖLÜM 5 — t-SNE Görselleştirme + Benzer Kelimeler
  7) BÖLÜM 6 — Sınıflandırma: TF-IDF vs Word2Vec vs FastText
  8) BÖLÜM 7 — Sonuç Tablosu ve Özet

Kaynak:
  https://huggingface.co/datasets/engin1123/keloglan-turkish-sentiment-analysis-dataset
  630K+ Türkçe yorum (Pozitif/Nötr/Negatif)

Çalıştırma:
  pip install -r requirements.txt
  python word_embeddings_kapsamli.py
"""

import os
import sys
import re
import math
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
# SABITLER
# ======================================================================
FIG_DIR = "figures"
RANDOM_STATE = 42
SAMPLE_SIZE = 50000       # Hız için 50K örnek (toplam 630K)
EMBEDDING_DIM = 100
plt.rcParams["figure.dpi"] = 120


# ======================================================================
# YARDIMCI
# ======================================================================
def baslik_yaz(metin):
    print("\n" + "=" * 70)
    print(metin)
    print("=" * 70)


# ======================================================================
# BÖLÜM 1 — TEORI
# ======================================================================
def bolum1_teori():
    baslik_yaz("BÖLÜM 1: WORD EMBEDDINGS TEORISI")
    print("""
One-Hot Encoding:
  Her kelime = [0, 0, ..., 1, ..., 0]  (10.000 boyutlu, sadece 1 tane 1)
  Sorun: "kral" ile "kralice" arasinda 0 benzerlik! Anlam yok.

Word Embeddings:
  Her kelime = [0.23, -0.45, 0.12, ..., 0.89]  (~100-300 boyutlu, yogun)
  Anlamca yakin kelimeler => vektor uzayinda yakin!
  "kral" ile "kralice" benzer => yuksek cosine similarity

Word2Vec (Mikolov et al., 2013):
  - Skip-gram: Hedef kelimeden komsu kelimeleri tahmin et
  - CBOW: Komsu kelimelerden hedef kelimeyi tahmin et

FastText (Bojanowski et al., 2016):
  - Word2Vec + subword (karakter n-gram'lari)
  - Bilinmeyen kelimeleri (OOV) de tahmin edebilir
  - Morfolojik olarak zengin dillerde (Türkçe gibi) daha iyi

GloVe (Pennington et al., 2014):
  - Kelime birlikte görülme (co-occurrence) istatistiklerine dayali
  - Matris faktorizasyonu + Word2Vec hibriti
""")


# ======================================================================
# BÖLÜM 2 — VERI YÜKLEME
# ======================================================================
def bolum2_veri_yukle():
    baslik_yaz("BÖLÜM 2: TÜRKÇE SENTIMENT VERI SETI YÜKLENIYOR")

    print("Kaynak: engin1123/keloglan-turkish-sentiment-analysis-dataset")
    print("(HuggingFace, 630.000+ Türkçe yorum, CC-BY-NC 4.0)")

    try:
        from datasets import load_dataset
        dataset = load_dataset("engin1123/keloglan-turkish-sentiment-analysis-dataset",
                                split="train", streaming=False)
        df = dataset.to_pandas()
    except Exception as e:
        print(f"[HATA] HuggingFace verisi yüklenemedi: {e}")
        print("[INFO] Basit örnek veri ile devam ediliyor...")
        # Fallback: sentetik Türkçe veri
        np.random.seed(RANDOM_STATE)
        pozitif = [
            "harika ürün çok beğendim kesinlikle tavsiye ederim",
            "mükemmel kalite ve hizmet herkese öneririm",
            "çok güzel ve kaliteli tam istediğim gibi",
            "fiyat performans ürünü kesinlikle harika",
            "çok memnun kaldım tekrar alacağım",
        ]
        negatif = [
            "berbat bir ürün hiç beğenmedim kesinlikle tavsiye etmiyorum",
            "kalitesiz ve kötü para iadesi istiyorum",
            "çok kötü kargoda hasarlı geldi hiç memnun kalmadım",
            "beklediğim gibi çıkmadı hayal kırıklığı",
            "rezalet bir alışveriş deneyimi yaşadım",
        ]
        notr = [
            "ürün normal beklentimi karşıladı ne iyi ne kötü",
            "ortalama bir ürün idare eder",
            "fiyatına göre normal sayılabilir",
            "beklediğim gibi çıktı sorun yok",
            "standart bir ürün herhangi bir özelliği yok",
        ]
        texts = pozitif * 4000 + negatif * 3000 + notr * 3000
        labels = [2]*len(pozitif*4000) + [0]*len(negatif*3000) + [1]*len(notr*3000)
        df = pd.DataFrame({"text": texts, "label": labels})
        print(f"[INFO] Fallback: {len(df)} örnek oluşturuldu.")

    # Etiket dağılımı
    print(f"\nToplam örnek: {len(df)}")
    print(f"Sütunlar: {list(df.columns)}")
    etiket_map = {0: "Negatif", 1: "Nötr", 2: "Pozitif"}
    print("\nEtiket dağılımı:")
    for l, name in etiket_map.items():
        count = (df["label"] == l).sum()
        print(f"  {name}: {count} (%{100*count/len(df):.1f})")

    # Örnek metinler
    print("\nÖrnek metinler:")
    for i in range(3):
        print(f"  [{etiket_map[df['label'].iloc[i]]}] {df['text'].iloc[i][:80]}...")

    # Hız için örnekleme
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"\nÖrnekleme: {SAMPLE_SIZE} metin kullanılacak.")

    # Metin temizleme
    def temizle(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zöçşığüâîû0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df["clean_text"] = df["text"].apply(temizle)

    # Tokenize
    df["tokens"] = df["clean_text"].apply(lambda x: x.split())

    print(f"\nÖrnek temizlenmiş metin:")
    print(f"  Ham:  {df['text'].iloc[0][:80]}")
    print(f"  Temiz: {df['clean_text'].iloc[0][:80]}")

    return df


# ======================================================================
# BÖLÜM 3 — WORD2VEC
# ======================================================================
def bolum3_word2vec(df):
    baslik_yaz("BÖLÜM 3: WORD2VEC EĞITIMI (gensim)")

    try:
        from gensim.models import Word2Vec
    except ImportError:
        print("[HATA] gensim yüklü değil. pip install gensim")
        return None, None

    sentences = df["tokens"].tolist()
    print(f"Toplam cümle: {len(sentences)}")
    print(f"Kelime dağarcığı oluşturuluyor...")

    model = Word2Vec(
        sentences=sentences,
        vector_size=EMBEDDING_DIM,
        window=5,
        min_count=5,
        workers=4,
        sg=1,         # Skip-gram (1) / CBOW (0)
        seed=RANDOM_STATE,
    )

    print(f"\nKelime dağarcığı: {len(model.wv)} kelime")
    print(f"Vektör boyutu: {model.wv.vector_size}")

    # Benzer kelime örnekleri (Türkçe)
    print("\nBenzer kelime örnekleri:")
    ornek_kelimeler = ["güzel", "kötü", "ürün", "fiyat", "harika"]
    for word in ornek_kelimeler:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=5)
            print(f"  '{word}' -> {[s[0] for s in similar]}")

    return model, sentences


# ======================================================================
# BÖLÜM 4 — FASTTEXT
# ======================================================================
def bolum4_fasttext(df):
    baslik_yaz("BÖLÜM 4: FASTTEXT EĞITIMI (gensim)")

    try:
        from gensim.models import FastText
    except ImportError:
        print("[HATA] gensim yüklü değil.")
        return None

    sentences = df["tokens"].tolist()
    model = FastText(
        sentences=sentences,
        vector_size=EMBEDDING_DIM,
        window=5,
        min_count=5,
        workers=4,
        sg=1,
        seed=RANDOM_STATE,
    )

    print(f"Kelime dağarcığı: {len(model.wv)} kelime")
    print(f"Vektör boyutu: {model.wv.vector_size}")

    # OOV testi: modelde olmayan bir kelime
    print("\nOOV (Out-of-Vocabulary) Testi:")
    test_words = ["güzellik", "kötülük", "ürüncük", "harikalı"]
    for word in test_words:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=3)
            print(f"  '{word}' (OOV) -> {[s[0] for s in similar]}")
        else:
            # FastText subword sayesinde OOV için vektör üretebilir
            print(f"  '{word}' (OOV) -> vektör var: {word in model.wv}")

    return model


# ======================================================================
# BÖLÜM 5 — t-SNE GÖRSELLESTIRME
# ======================================================================
def bolum5_tsne_ve_analoji(model, model_ft, df):
    baslik_yaz("BÖLÜM 5: t-SNE GÖRSELLESTIRME VE ANALOJI TESTLERI")

    os.makedirs(FIG_DIR, exist_ok=True)

    # t-SNE ile kelime vektörlerini 2 boyuta indirge
    print("t-SNE ile kelime vektörleri 2 boyuta indirgeniyor...")
    kelimeler = ["güzel", "harika", "mükemmel", "iyi", "süper",
                 "kötü", "berbat", "çirkin", "pis", "rezalet",
                 "fiyat", "para", "kalite", "ürün", "hizmet",
                 "hızlı", "yavaş", "büyük", "küçük", "ucuz",
                 "pahalı", "temiz", "kirli", "kolay", "zor"]

    # Modelde var olan kelimeleri filtrele
    if model is not None:
        kelimeler = [w for w in kelimeler if w in model.wv]
        vektorler = np.array([model.wv[w] for w in kelimeler])

        tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=5)
        xy = tsne.fit_transform(vektorler)

        plt.figure(figsize=(12, 10))
        pozitif_kel = ["güzel", "harika", "mükemmel", "iyi", "süper", "hızlı", "temiz", "kolay", "kalite"]
        for i, word in enumerate(kelimeler):
            renk = "green" if word in pozitif_kel else "red"
            plt.scatter(xy[i, 0], xy[i, 1], c=renk, s=100, alpha=0.7)
            plt.annotate(word, (xy[i, 0], xy[i, 1]), fontsize=10, ha='center',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

        plt.title("Word2Vec: Türkçe Kelime Vektörleri (t-SNE)", fontsize=14)
        plt.xlabel("t-SNE Bileşen 1")
        plt.ylabel("t-SNE Bileşen 2")
        plt.grid(True, alpha=0.3)
        out = os.path.join(FIG_DIR, "02_tsne_word2vec.png")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"[görsel] t-SNE -> {out}")

    # Analoji testi
    print("\nAnaloji Testleri (Word2Vec):")
    if model is not None:
        analojiler = [
            ("kral", "kraliçe", "adam"),
            ("iyi", "kötü", "güzel"),
            ("sıcak", "soğuk", "büyük"),
        ]
        for a, b, c in analojiler:
            if all(w in model.wv for w in [a, b, c]):
                try:
                    sonuc = model.wv.most_similar(positive=[b, c], negative=[a], topn=1)
                    print(f"  '{a}' - '{b}' = '{c}' - ? -> '{sonuc[0][0]}' ({sonuc[0][1]:.3f})")
                except:
                    pass

    return model


# ======================================================================
# BÖLÜM 6 — SINIFLANDIRMA KARŞILAŞTIRMASI
# ======================================================================
def bolum6_siniflandirma(df, model_w2v, model_ft):
    baslik_yaz("BÖLÜM 6: SINIFLANDIRMA KARŞILAŞTIRMASI")

    X = df["clean_text"].to_numpy()
    y = df["label"].to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    print(f"Eğitim: {len(X_tr)} | Test: {len(X_te)}")
    print()

    sonuclar = []

    # --- 6.1 TF-IDF + Logistic Regression ---
    print("[1/3] TF-IDF + Logistic Regression çalışıyor...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words=None)
    X_tr_tfidf = tfidf.fit_transform(X_tr)
    X_te_tfidf = tfidf.transform(X_te)

    clf_tfidf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf_tfidf.fit(X_tr_tfidf, y_tr)
    pred_tfidf = clf_tfidf.predict(X_te_tfidf)
    acc_tfidf = accuracy_score(y_te, pred_tfidf)
    f1_tfidf = f1_score(y_te, pred_tfidf, average='weighted')
    sonuclar.append(("TF-IDF", tfidf.get_feature_names_out().shape[0], acc_tfidf, f1_tfidf))
    print(f"  TF-IDF: Acc={acc_tfidf:.4f}  F1={f1_tfidf:.4f}")

    # --- 6.2 Word2Vec + Logistic Regression ---
    print("[2/3] Word2Vec + Logistic Regression çalışıyor...")
    if model_w2v is not None:
        def dokuman_vektoru(tokens, model):
            vektorler = [model.wv[w] for w in tokens if w in model.wv]
            if len(vektorler) == 0:
                return np.zeros(model.wv.vector_size)
            return np.mean(vektorler, axis=0)

        X_tr_w2v = np.array([dokuman_vektoru(t.split(), model_w2v) for t in X_tr])
        X_te_w2v = np.array([dokuman_vektoru(t.split(), model_w2v) for t in X_te])

        clf_w2v = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        clf_w2v.fit(X_tr_w2v, y_tr)
        pred_w2v = clf_w2v.predict(X_te_w2v)
        acc_w2v = accuracy_score(y_te, pred_w2v)
        f1_w2v = f1_score(y_te, pred_w2v, average='weighted')
        sonuclar.append(("Word2Vec", EMBEDDING_DIM, acc_w2v, f1_w2v))
        print(f"  Word2Vec: Acc={acc_w2v:.4f}  F1={f1_w2v:.4f}")

    # --- 6.3 FastText + Logistic Regression ---
    print("[3/3] FastText + Logistic Regression çalışıyor...")
    if model_ft is not None:
        def dokuman_vektoru_ft(tokens, model):
            vektorler = [model.wv[w] for w in tokens if w in model.wv]
            if len(vektorler) == 0:
                return np.zeros(model.wv.vector_size)
            return np.mean(vektorler, axis=0)

        X_tr_ft = np.array([dokuman_vektoru_ft(t.split(), model_ft) for t in X_tr])
        X_te_ft = np.array([dokuman_vektoru_ft(t.split(), model_ft) for t in X_te])

        clf_ft = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        clf_ft.fit(X_tr_ft, y_tr)
        pred_ft = clf_ft.predict(X_te_ft)
        acc_ft = accuracy_score(y_te, pred_ft)
        f1_ft = f1_score(y_te, pred_ft, average='weighted')
        sonuclar.append(("FastText", EMBEDDING_DIM, acc_ft, f1_ft))
        print(f"  FastText: Acc={acc_ft:.4f}  F1={f1_ft:.4f}")

    # Sonuç tablosu
    print("\n" + "=" * 60)
    print("SONUÇ KARŞILAŞTIRMA TABLOSU")
    print("=" * 60)
    print(f"{'Yöntem':12s} | {'Boyut':8s} | {'Accuracy':10s} | {'F1':10s}")
    print("-" * 60)
    for name, dim, acc, f1 in sonuclar:
        print(f"{name:12s} | {dim:<8d} | {acc:<10.4f} | {f1:<10.4f}")

    # Görsel: karşılaştırma
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    isimler = [s[0] for s in sonuclar]
    accs = [s[2] for s in sonuclar]
    f1s = [s[3] for s in sonuclar]
    x = np.arange(len(isimler))
    w = 0.35
    ax.bar(x - w/2, accs, w, label='Accuracy', color='#0275d8')
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
    out = os.path.join(FIG_DIR, "03_classification_comparison.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[görsel] Sınıflandırma karşılaştırması -> {out}")

    return sonuclar


# ======================================================================
# BÖLÜM 7 — ÖZET
# ======================================================================
def bolum7_ozet():
    baslik_yaz("BÖLÜM 7: ÖZET VE KARŞILAŞTIRMA")
    print("""
ÖZET: Word Embeddings vs TF-IDF
=================================

TF-IDF:
  + Basit, hızlı, yorumlanabilir
  + Seyrek vektörler (sparse)
  + Stop-word'leri otomatik düşürür
  - Kelime anlamını bilmez
  - Bag-of-words (kelime sırası yok)

Word Embeddings (Word2Vec / FastText):
  + Kelimelerin anlamını öğrenir
  + Yoğun vektörler (dense, ~100-300 boyut)
  + Cosine similarity anlamlıdır
  + FastText: OOV kelimeleri de tahmin edebilir
  - Daha fazla veri gerekir
  - Eğitimi daha yavaş
  - Yorumlaması zor

Ne Zaman Hangisi?
  - TF-IDF: Küçük veri, hızlı prototip, baseline
  - Word2Vec: Orta-büyük veri, anlamsal benzerlik önemliyse
  - FastText: Türkçe gibi sondan eklemeli dillerde
  - BERT: En iyi performans (ama çok daha yavaş)
""")


# ======================================================================
# ANA AKIŞ
# ======================================================================
def main():
    print("=" * 70)
    print("  WORD EMBEDDINGS: WORD2VEC, FASTTEXT ve TF-IDF")
    print("  Kapsamli Anlatim, Uygulama ve Analiz")
    print("=" * 70)

    bolum1_teori()
    df = bolum2_veri_yukle()
    model_w2v, sentences = bolum3_word2vec(df)
    model_ft = bolum4_fasttext(df)
    bolum5_tsne_ve_analoji(model_w2v, model_ft, df)
    sonuclar = bolum6_siniflandirma(df, model_w2v, model_ft)
    bolum7_ozet()

    print("\n" + "=" * 70)
    print("TAMAMLANDI")
    print(f"Tüm görseller 'figures/' klasörüne kaydedildi.")
    print("=" * 70)


if __name__ == "__main__":
    main()
