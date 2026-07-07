#!/usr/bin/env python3
"""
tfidf_kapsamli.py
=================
TF-IDF: Teoriden Pratiğe — Manuel Hesaplama, Scikit-learn, Sınıflandırma ve SVD Analizi

Akış:
  1) Kütüphaneler ve sabitler
  2) BÖLÜM 1 — Teori: TF-IDF bileşenleri
  3) BÖLÜM 2 — Manuel Hesaplama: 4 Türkçe dokümanla adım adım TF-IDF
  4) BÖLÜM 3 — Scikit-learn ile TF-IDF + Görselleştirme
  5) BÖLÜM 4 — 20 Newsgroups: Gerçek veri + Cosine Similarity
  6) BÖLÜM 5 — Sınıflandırma: TF-IDF + Logistic Regression
  7) BÖLÜM 6 — SVD ile Boyut İndirgeme ve Performans Takası
  8) BÖLÜM 7 — TF-IDF Limitleri (Semantik, Bağlam, Seyreklik, Boyut Laneti)
  9) BÖLÜM 8 — Örnek Tahminler ve Özet

Çalıştırma:
  python tfidf_kapsamli.py

Üretilenler:
  figures/ -> tüm grafikler (PNG)
"""

import math
import os
import sys
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from scipy.spatial.distance import pdist

# ======================================================================
# SABITLER
# ======================================================================
FIG_DIR = "figures"
RANDOM_STATE = 42
plt.rcParams["figure.dpi"] = 120


# ======================================================================
# YARDIMCI FONKSIYONLAR
# ======================================================================
def baslik_yaz(metin):
    print("\n" + "=" * 70)
    print(metin)
    print("=" * 70)


# ======================================================================
# BÖLÜM 1 — TEORI
# ======================================================================
def bolum1_teori():
    baslik_yaz("BÖLÜM 1: TF-IDF TEORISI VE BILESENLERI")
    print("""
TF-IDF = Term Frequency x Inverse Document Frequency

TF (Term Frequency) - Terim Sikligi:
    Bir terimin bir dokümanda ne siklikta geçtigini gösterir.
    TF(t,d) = terimin dokümanda geçme sayisi / dokümandaki toplam terim sayisi

IDF (Inverse Document Frequency) - Ters Doküman Sikligi:
    Bir terimin tüm dokümanlarda ne kadar nadir geçtigini ölçer.
    IDF(t) = log( toplam doküman sayisi / terimi içeren doküman sayisi )

TF-IDF:
    TF-IDF(t,d) = TF(t,d) x IDF(t)

Yorum:
  - TF yüksek  = kelime bu dokümanda önemli
  - IDF yüksek = kelime genelde nadir (ayirt edici)
  - TF-IDF yüksek = kelime bu doküman için karakteristik
""")


# ======================================================================
# BÖLÜM 2 — MANUEL HESAPLAMA
# ======================================================================
def bolum2_manuel_hesaplama():
    baslik_yaz("BÖLÜM 2: ADIM ADIM MANUEL TF-IDF HESAPLAMA")

    documents = [
        "kedi çok tatli bir hayvan",      # D1
        "kedi ve köpek en iyi arkadas",   # D2
        "köpek sadik bir hayvandir",      # D3
        "araba hizli gider",              # D4
    ]
    N = len(documents)
    all_words = sorted(set(w for d in documents for w in d.split()))

    print(f"Dokümanlar ({N} adet):")
    for i, d in enumerate(documents):
        print(f"  D{i+1}: \"{d}\"")
    print(f"\nTüm benzersiz kelimeler: {all_words}\n")

    # Adim adim hesaplama tablosu
    print("-" * 130)
    print(f"{'Kelime':12s} | {'D1 TF':8s} {'D1 IDF':8s} {'D1 TF-IDF':10s} | "
          f"{'D2 TF':8s} {'D2 IDF':8s} {'D2 TF-IDF':10s} | "
          f"{'D3 TF':8s} {'D3 IDF':8s} {'D3 TF-IDF':10s} | "
          f"{'D4 TF':8s} {'D4 IDF':8s} {'D4 TF-IDF':10s} | "
          f"{'Iceren Doc':10s} {'IDF':8s}")
    print("-" * 130)

    for word in all_words:
        doc_containing = sum(1 for d in documents if word in d.split())
        idf_val = math.log((N + 1) / (doc_containing + 1)) + 1

        row = f"{word:12s} |"
        for d in documents:
            words_in_doc = d.split()
            tf = words_in_doc.count(word) / len(words_in_doc)
            tfidf = tf * idf_val
            row += f" {tf:<8.3f} {idf_val:<8.3f} {tfidf:<10.4f} |"

        row += f" d={doc_containing:<3d}     {idf_val:<8.3f}"
        print(row)

    print("-" * 130)

    # "kedi" özel hesabi
    print("\n--- 'kedi' KELIMESI IÇIN ADIM ADIM HESAP ---")
    word = "kedi"
    dc = sum(1 for d in documents if word in d.split())
    idf = math.log((N + 1) / (dc + 1)) + 1
    print(f"  IDF('kedi') = log(({N}+1)/({dc}+1)) + 1 = {idf:.3f}")
    print(f"  D1'de TF-IDF('kedi') = (1/5) x {idf:.3f} = {(1/5*idf):.4f}")
    print(f"  D2'de TF-IDF('kedi') = (1/6) x {idf:.3f} = {(1/6*idf):.4f}")

    # Nadir vs sik karsilastirmasi
    print("\n--- NADIR vs SIK KELIME KARSILASTIRMASI ---")
    for word in ["araba", "ve", "kedi"]:
        dc = sum(1 for d in documents if word in d.split())
        idf_val = math.log((N + 1) / (dc + 1)) + 1
        etiket = "NADIR (ayirt edici)" if idf_val > 1.7 else "SIK (az ayirt edici)"
        print(f"  '{word}': {dc}/{N} dokümanda -> IDF = {idf_val:.3f} -> {etiket}")

    # Küçük dokümanlar ile TF-IDF
    print("\n--- KÜÇÜK DOKÜMANLAR ILE TF-IDF (sifirdan) ---")
    kucuk_dokumanlar = [
        "kedi evde uyuyor",
        "köpek parkta kosuyor",
        "kedi ve köpek birlikte oynuyor",
        "evde kedi mamasi var",
    ]
    tokenize_kucuk = [d.lower().split() for d in kucuk_dokumanlar]

    def tf(term, tokenize_doc):
        count = tokenize_doc.count(term)
        return count / len(tokenize_doc) if len(tokenize_doc) > 0 else 0

    def idf(term, tokenize_dokumanlar):
        doc_count = sum(1 for doc in tokenize_dokumanlar if term in doc)
        return math.log(len(tokenize_dokumanlar) / (1 + doc_count)) + 1

    for i, doc in enumerate(tokenize_kucuk):
        print(f"  --- D{i+1}: '{kucuk_dokumanlar[i]}' ---")
        for term in sorted(set(doc)):
            tf_val = tf(term, doc)
            idf_val = idf(term, tokenize_kucuk)
            tfidf_val = tf_val * idf_val
            print(f"    '{term}' -> TF={tf_val:.4f}  IDF={idf_val:.4f}  TF-IDF={tfidf_val:.4f}")


# ======================================================================
# BÖLÜM 3 — SCIKIT-LEARN ILE TF-IDF
# ======================================================================
def bolum3_scikit_learn():
    baslik_yaz("BÖLÜM 3: SCIKIT-LEARN ILE TF-IDF + GÖRSELLESTIRME")

    kucuk_dokumanlar = [
        "kedi evde uyuyor",
        "köpek parkta kosuyor",
        "kedi ve köpek birlikte oynuyor",
        "evde kedi mamasi var",
    ]

    vectorizer = TfidfVectorizer()
    tfidf_matrisi = vectorizer.fit_transform(kucuk_dokumanlar)

    df = pd.DataFrame(
        tfidf_matrisi.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=[f"Doküman {i+1}" for i in range(len(kucuk_dokumanlar))]
    )

    print("\nTF-IDF Matrisi (DataFrame):")
    print(df.to_string())
    print()

    print("Her dokümanda en yüksek TF-IDF skoruna sahip kelime:")
    for i, doc in enumerate(kucuk_dokumanlar):
        satir = df.iloc[i]
        en_iyi = satir.idxmax()
        skor = satir.max()
        print(f"  D{i+1}: '{doc}' -> '{en_iyi}' (skor={skor:.4f})")

    # Heatmap
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap="YlOrRd", fmt=".3f", linewidths=0.5)
    plt.title("TF-IDF Matrisi (Isi Haritasi)")
    plt.xlabel("Kelimeler")
    plt.ylabel("Dokümanlar")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "01_tfidf_heatmap.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[görsel] Isi haritasi kaydedildi -> {out}")


# ======================================================================
# BÖLÜM 4 — 20 NEWSGROUPS
# ======================================================================
def bolum4_newsgroups():
    baslik_yaz("BÖLÜM 4: 20 NEWSGROUPS ILE GERÇEK VERI UYGULAMASI")

    kategoriler = ["rec.sport.baseball", "sci.space", "comp.graphics", "talk.politics.guns"]
    print(f"Kategoriler: {kategoriler}")

    newsgroups = fetch_20newsgroups(subset='train', categories=kategoriler,
                                    shuffle=True, random_state=RANDOM_STATE)
    print(f"Yüklenen belge sayisi: {len(newsgroups.data)}")

    vectorizer_news = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer_news.fit_transform(newsgroups.data)

    print(f"TF-IDF matris boyutu: {X.shape}")
    print(f"Seyreklik: %{(1 - X.nnz / (X.shape[0] * X.shape[1])) * 100:.2f}")

    # Cosine similarity
    ilk_kategori = kategoriler[0]
    kategori_belgeler = [i for i, t in enumerate(newsgroups.target)
                         if t == kategoriler.index(ilk_kategori)]
    secili = kategori_belgeler[:3]

    benzerlikler = cosine_similarity(X[secili[0]:secili[0]+1],
                                      X[secili[1]:secili[2]+1])[0]
    print(f"\nCosine Similarity (kategori: {ilk_kategori}):")
    print(f"  Belge 1 vs Belge 2: {benzerlikler[0]:.4f}")
    print(f"  Belge 1 vs Belge 3: {benzerlikler[1]:.4f}")

    # Farkli kategori
    farkli_kategori = kategoriler[1]
    farkli_indeks = [i for i, t in enumerate(newsgroups.target)
                     if t == kategoriler.index(farkli_kategori)][0]
    benzerlik_farkli = cosine_similarity(X[secili[0]:secili[0]+1],
                                          X[farkli_indeks:farkli_indeks+1])[0][0]
    print(f"\nAyni kategori (baseball) benzerligi: {benzerlikler[0]:.4f}")
    print(f"Farkli kategori (space) benzerligi:   {benzerlik_farkli:.4f}")
    print("-> Ayni kategorideki belgeler arasi benzerlik daha yüksektir.")

    return newsgroups, vectorizer_news


# ======================================================================
# BÖLÜM 5 — SINIFLANDIRMA
# ======================================================================
def bolum5_siniflandirma():
    baslik_yaz("BÖLÜM 5: TF-IDF + LOGISTIC REGRESSION ILE SINIFLANDIRMA")

    kategoriler = ["rec.sport.baseball", "sci.space", "comp.graphics", "talk.politics.guns"]
    newsgroups = fetch_20newsgroups(subset='all', categories=kategoriler,
                                    shuffle=True, random_state=RANDOM_STATE)
    print(f"Toplam belge: {len(newsgroups.data)}")

    X = newsgroups.data
    y = newsgroups.target

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f"Egitim: {len(X_tr)} | Test: {len(X_te)}")

    MAX_FEATURES = 10000
    model = make_pipeline(
        TfidfVectorizer(stop_words='english', ngram_range=(1, 2),
                         min_df=5, max_features=MAX_FEATURES, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight='balanced',
                           random_state=RANDOM_STATE),
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    acc = accuracy_score(y_te, pred)
    f1m = f1_score(y_te, pred, average='macro')
    n_feat = len(model.named_steps['tfidfvectorizer'].get_feature_names_out())
    clf = model.named_steps['logisticregression']
    vec = model.named_steps['tfidfvectorizer']

    print(f"\nÖznitelik sayisi: {n_feat}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (macro): {f1m:.4f}")
    print("\nSiniflandirma Raporu:")
    print(classification_report(y_te, pred, target_names=kategoriler))

    # Confusion matrix
    os.makedirs(FIG_DIR, exist_ok=True)
    cm = confusion_matrix(y_te, pred)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(kategoriler)), labels=[k.split('.')[-1] for k in kategoriler],
                  rotation=30, ha='right')
    ax.set_yticks(range(len(kategoriler)), labels=[k.split('.')[-1] for k in kategoriler])
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title(f"Confusion Matrix (TF-IDF + LogReg) — Acc=%{acc*100:.1f}")
    for i in range(len(kategoriler)):
        for j in range(len(kategoriler)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "02_confusion_matrix.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[görsel] Confusion matrix -> {out}")

    # En etkili kelimeler (top words)
    isimler = np.array(vec.get_feature_names_out())
    katsayilar = clf.coef_[0]
    n_top = 12
    top_pos = np.argsort(katsayilar)[-n_top:]
    top_neg = np.argsort(katsayilar)[:n_top]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(isimler[top_pos], katsayilar[top_pos], color="#5cb85c")
    axes[0].set_title("En olumlu etki (baseball lehine)")
    axes[1].barh(isimler[top_neg], katsayilar[top_neg], color="#d9534f")
    axes[1].set_title("En olumsuz etki (baseball aleyhine)")
    for ax in axes:
        ax.axvline(0, color="gray", lw=0.8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "03_top_words.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[görsel] En etkili kelimeler -> {out}")

    return acc, f1m, n_feat


# ======================================================================
# BÖLÜM 6 — SVD ANALIZI
# ======================================================================
def bolum6_svd_analizi():
    baslik_yaz("BÖLÜM 6: SVD ILE BOYUT INDIRGEME VE PERFORMANS TAKASI")

    kategoriler = ["rec.sport.baseball", "sci.space", "comp.graphics", "talk.politics.guns"]
    newsgroups = fetch_20newsgroups(subset='all', categories=kategoriler,
                                    shuffle=True, random_state=RANDOM_STATE)
    X = newsgroups.data
    y = newsgroups.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    MAX_FEATURES = 10000
    SVD_COMPONENTS = [25, 50, 100, 200, 500]

    accs, f1s, evrs = [], [], []
    for k in SVD_COMPONENTS:
        pipe = make_pipeline(
            TfidfVectorizer(stop_words='english', ngram_range=(1, 2),
                             min_df=5, max_features=MAX_FEATURES, sublinear_tf=True),
            TruncatedSVD(n_components=k, random_state=RANDOM_STATE),
            LogisticRegression(max_iter=1000, class_weight='balanced',
                               random_state=RANDOM_STATE),
        )
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        accs.append(accuracy_score(y_te, pred))
        f1s.append(f1_score(y_te, pred, average='macro'))
        evrs.append(pipe.named_steps['truncatedsvd'].explained_variance_ratio_.sum())
        print(f"  SVD k={k:<4} -> acc={accs[-1]:.4f}  F1={f1s[-1]:.4f}  varyans=%{100*evrs[-1]:.1f}")

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    ax1.plot(SVD_COMPONENTS, accs, "o-", color="#0275d8", label="Accuracy")
    ax1.plot(SVD_COMPONENTS, f1s, "s-", color="#5cb85c", label="F1 (macro)")
    ax1.set_xlabel("SVD bilesen sayisi (boyut)")
    ax1.set_ylabel("Skor")
    ax1.legend(loc="lower right")
    ax2 = ax1.twinx()
    ax2.plot(SVD_COMPONENTS, [100*e for e in evrs], "^--", color="#f0ad4e",
             label="Açiklanan varyans %")
    ax2.set_ylabel("Açiklanan varyans (%)", color="#f0ad4e")
    ax1.set_title("SVD: Boyut Indirgeme — Performans Takasi")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "04_svd_tradeoff.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[görsel] SVD takas grafigi -> {out}")

    return accs, f1s, evrs


# ======================================================================
# BÖLÜM 7 — TF-IDF LIMITLERI
# ======================================================================
def bolum7_limitler():
    baslik_yaz("BÖLÜM 7: TF-IDF LIMITLERI")

    # 7.1 Semantik
    print("\n--- 7.1 Anlamsal Iliski (Semantics) ---")
    print("TF-IDF her kelimeyi bagimsiz bir ID olarak ele alir.")
    print('"Kral" ile "Kraliçe" arasinda semantik iliskiyi bilmez.\n')

    dok_semantik = [
        "kral sarayda yasar",
        "kralice sarayda yasar",
        "masa odada duruyor",
    ]
    v = TfidfVectorizer()
    m = v.fit_transform(dok_semantik)
    benzerlik = cosine_similarity(m[0:1], m[1:3])[0]
    print(f"  'kral ...' vs 'kralice ...': {benzerlik[0]:.4f}")
    print(f"  'kral ...' vs 'masa ...'   : {benzerlik[1]:.4f}")
    print("  Iki deger neredeyse ayni! Oysa kral ve kralice anlamsal olarak çok daha yakin.")

    # 7.2 Baglam
    print("\n--- 7.2 Baglam (Context) ---")
    print('TF-IDF bag-of-words modelidir. "Köpek adami isirdi" ile "Adam köpegi isirdi"')
    print('TF-IDF için neredeyse aynidir, oysa anlamlari tamamen farklidir!\n')

    dok_baglam = [
        "köpek adami isirdi",
        "adam köpegi isirdi",
        "kedi fareyi kovaladi",
    ]
    v3 = TfidfVectorizer()
    m3 = v3.fit_transform(dok_baglam)
    b2 = cosine_similarity(m3[0:1], m3[1:3])[0]
    print(f"  unigram benzerlik: {b2[0]:.4f} (neredeyse özdes!)")

    # N-gram kisni çözüm
    v4 = TfidfVectorizer(ngram_range=(1, 2))
    m4 = v4.fit_transform(dok_baglam)
    b3 = cosine_similarity(m4[0:1], m4[1:3])[0]
    print(f"  bigram benzerlik:   {b3[0]:.4f} (daha farkli, ama yetersiz)")
    print("  N-gram'lar kisni çözüm sunar. Gerçek çözüm: RNN, LSTM, Transformer.")

    # 7.3 Seyreklik
    print("\n--- 7.3 Boyutlanabilirlik ve Seyreklik ---")
    np.random.seed(RANDOM_STATE)
    temel_kelimeler = [
        "teknoloji", "yapay", "zeka", "veri", "bilim", "yazilim", "makine",
        "ögrenme", "derin", "sinir", "dil", "isleme", "analiz", "model",
        "algoritma", "optimizasyon", "matris", "siniflandirma", "tahmin",
        "kral", "kralice", "masa", "kitap", "spor", "ekonomi", "saglik",
        "egitim", "hukuk", "sanat", "muzik", "tarih", "politika", "cevre",
        "doga", "enerji", "dijital", "robot", "uzay", "zaman",
    ]

    NUM_DOCS = 1000
    buyuk_dokumanlar = []
    for _ in range(NUM_DOCS):
        doc_len = np.random.randint(20, 100)
        doc = " ".join(np.random.choice(temel_kelimeler, doc_len, replace=True))
        buyuk_dokumanlar.append(doc)

    v5 = TfidfVectorizer()
    m5 = v5.fit_transform(buyuk_dokumanlar)
    toplam_hucre = m5.shape[0] * m5.shape[1]
    seyreklik = 100 * (1 - m5.nnz / toplam_hucre)
    print(f"  Doküman sayisi: {m5.shape[0]}")
    print(f"  Kelime dagarcigi: {m5.shape[1]}")
    print(f"  Seyreklik: %{seyreklik:.2f}")
    print(f"  (Her dokümanda ortalama {m5.nnz/m5.shape[0]:.1f} sifir olmayan öznitelik)")

    # Seyreklik grafigi
    max_features_degerleri = [100, 500, 1000, 2000, 5000]
    seyreklikler, boyutlar = [], []
    for mf in max_features_degerleri:
        v = TfidfVectorizer(max_features=mf)
        m = v.fit_transform(buyuk_dokumanlar)
        toplam = m.shape[0] * m.shape[1]
        sey = 100 * (1 - m.nnz / toplam)
        seyreklikler.append(sey)
        boyutlar.append(m.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(boyutlar, seyreklikler, marker='o', linewidth=2)
    axes[0].set_xlabel("Vocabulary Boyutu")
    axes[0].set_ylabel("Seyreklik (%)")
    axes[0].set_title("Vocabulary Boyutu vs Seyreklik")
    axes[0].grid(True)
    axes[0].set_ylim(80, 100)

    axes[1].bar([str(b) for b in boyutlar], seyreklikler, color='coral')
    axes[1].set_xlabel("Vocabulary Boyutu")
    axes[1].set_ylabel("Seyreklik (%)")
    axes[1].set_title("Vocabulary Büyüdükçe Seyreklik Artar")
    axes[1].set_ylim(80, 100)
    for i, v in enumerate(seyreklikler):
        axes[1].text(i, v + 1, f"%{v:.1f}", ha="center", fontsize=9)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "05_sparsity.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[görsel] Seyreklik analizi -> {out}")

    # 7.4 Boyut Laneti
    print("\n--- 7.4 Boyut Laneti (Curse of Dimensionality) ---")
    print("Yüksek boyutta tüm noktalar birbirine esit uzaklikta görünür.")
    print("CV (Std/Ort) 0'a yaklasir -> mesafeler anlamsizlasir.\n")

    boyutlar = [2, 5, 10, 50, 100, 500]
    noktalar = 50
    mesafe_ort, mesafe_std = [], []
    for dim in boyutlar:
        data = np.random.uniform(0, 1, (noktalar, dim))
        mesafeler = pdist(data, 'euclidean')
        mesafe_ort.append(np.mean(mesafeler))
        mesafe_std.append(np.std(mesafeler))

    cv = [s / m for m, s in zip(mesafe_ort, mesafe_std)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(boyutlar, mesafe_ort, marker='o')
    axes[0].fill_between(boyutlar,
                         np.array(mesafe_ort) - np.array(mesafe_std),
                         np.array(mesafe_ort) + np.array(mesafe_std),
                         alpha=0.2)
    axes[0].set_xlabel("Boyut")
    axes[0].set_ylabel("Öklid Mesafesi")
    axes[0].set_title("Boyut Arttikça Tüm Noktalar Birbirine Uzaklasir")
    axes[0].grid(True)

    axes[1].plot(boyutlar, cv, marker='o', color='red')
    axes[1].axhline(y=0, color='gray', linestyle='--')
    axes[1].set_xlabel("Boyut")
    axes[1].set_ylabel("CV (Std / Ort)")
    axes[1].set_title("Boyut Arttikça Mesafeler Ayirt Edilemez")
    axes[1].grid(True)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "06_curse_of_dimensionality.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[görsel] Boyut laneti -> {out}")


# ======================================================================
# BÖLÜM 8 — ÖZET VE ÖRNEK TAHMIN
# ======================================================================
def bolum8_ozet():
    baslik_yaz("BÖLÜM 8: ÖZET VE ÖRNEK TAHMINLER")
    print("""
TF-IDF ÖZETI:
  TF-IDF = Term Frequency x Inverse Document Frequency
  - TF: kelimenin dokümanda ne kadar geçtigi
  - IDF: kelimenin tüm dokümanlarda ne kadar nadir oldugu
  - TF-IDF: ikisinin çarpimi -> kelimenin doküman için önemi

TF-IDF NE ZAMAN KULLANILIR?
  - Hizli ve basit çözüm gerektiginde
  - Kelime sirasinin kritik olmadigi görevlerde (spam tespiti, konu siniflandirma)
  - Küçük-orta ölçekli veri setlerinde
  - Ilk prototip / baseline model olarak

TF-IDF SINIRLAMALARI VE ÇÖZÜMLERI:
  | Sinirlama        | Sorun                              | Çözüm                     |
  |------------------|------------------------------------|---------------------------|
  | Anlamsal Iliski  | Es anlamlilar farkli ID'ler        | Word2Vec, GloVe, FastText |
  | Baglam Kaybi     | Kelime sirasi yok sayilir          | RNN, LSTM, Transformer    |
  | Seyreklik        | Vektörlerin %99+ sifir             | Embedding, SVD            |
  | Boyut Laneti     | Yüksek boyutta mesafe anlamsiz     | PCA, t-SNE, UMAP          |
""")

    print("ÖRNEK TAHMIN (20 Newsgroups siniflandirici):")
    kategoriler = ["rec.sport.baseball", "sci.space", "comp.graphics", "talk.politics.guns"]
    newsgroups = fetch_20newsgroups(subset='all', categories=kategoriler,
                                    shuffle=True, random_state=RANDOM_STATE)

    ornek_metinler = [
        "I love watching baseball games at the stadium with my family",
        "The rocket launched successfully and reached orbit",
        "The government should pass stricter gun control laws",
        "I need help with rendering 3D graphics in Python",
    ]

    model = make_pipeline(
        TfidfVectorizer(stop_words='english', ngram_range=(1, 2),
                         min_df=5, max_features=10000, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight='balanced',
                           random_state=RANDOM_STATE),
    )
    model.fit(newsgroups.data, newsgroups.target)

    for metin in ornek_metinler:
        p = model.predict([metin])[0]
        prob = model.predict_proba([metin])[0]
        en_iyi_idx = np.argmax(prob)
        print(f"  '{metin[:50]}...'")
        print(f"    -> {kategoriler[en_iyi_idx]} (%{prob[en_iyi_idx]*100:.1f})")


# ======================================================================
# ANA AKIS
# ======================================================================
def main():
    print("=" * 70)
    print("  TF-IDF: TEORIDEN PRATIGE")
    print("  Kapsamli Anlatim, Uygulama ve Analiz")
    print("=" * 70)

    bolum1_teori()
    bolum2_manuel_hesaplama()
    bolum3_scikit_learn()
    bolum4_newsgroups()
    acc, f1m, n_feat = bolum5_siniflandirma()
    accs, f1s, evrs = bolum6_svd_analizi()
    bolum7_limitler()
    bolum8_ozet()

    print("\n" + "=" * 70)
    print("SONUÇ TABLOSU")
    print("=" * 70)
    print(f"TF-IDF + Logistic Regression:  {n_feat} boyut -> Acc={acc:.4f}  F1={f1m:.4f}")
    for i, k in enumerate([25, 50, 100, 200, 500]):
        print(f"TF-IDF + SVD({k:3d}) + LogReg:    {k:5d} boyut -> Acc={accs[i]:.4f}  F1={f1s[i]:.4f}  varyans=%{100*evrs[i]:.1f}")

    print(f"\nTüm görseller 'figures/' klasörüne kaydedildi.")
    print("=" * 70)


if __name__ == "__main__":
    main()
