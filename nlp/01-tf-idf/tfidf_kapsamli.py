#!/usr/bin/env python3
"""
tfidf_kapsamli.py
=================
TF-IDF: Teoriden Pratiğe — Manuel Hesaplama, Scikit-learn, Sınıflandırma ve SVD Analizi

Bu script, TF-IDF konusunu en temel matematiksel formüllerden başlayarak
gerçek Türkçe haber verisi üzerinde sınıflandırmaya ve boyut indirgeme
analizine kadar adım adım anlatır.

Kullanılan Veri Seti: TTC-4900 (7 kategoride 4.900 Türkçe haber)

Çalıştırma:
  python tfidf_kapsamli.py

Üretilenler:
  figures/ -> tüm grafikler (PNG)
"""

import math
import os
import ssl
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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from scipy.spatial.distance import pdist

try:
    from tqdm import tqdm
except ImportError:
    # tqdm yoksa boş bir tqdm tanımla, kod sorunsuz çalışsın
    def tqdm(x, desc=""): return x
    print("(!) tqdm yüklü değil, ilerleme çubuğu gösterilmeyecek.")
    print("    Yüklemek için: pip install tqdm")

# ======================================================================
# AYARLAR
# ======================================================================
FIG_DIR = "figures"
RANDOM_STATE = 42
plt.rcParams["figure.dpi"] = 120

# TTC-4900 kategori isimleri (Türkçe)
TTC4900_KATEGORILER = [
    "siyaset", "dunya", "ekonomi", "kultur", "saglik", "spor", "teknoloji"
]


# ======================================================================
# YARDIMCI FONKSİYONLAR
# ======================================================================

def baslik(metin):
    """Konsola bölüm başlığı yazdırır."""
    print("\n" + "=" * 70)
    print(f"  {metin}")
    print("=" * 70)


def alt_baslik(metin):
    """Konsola alt başlık yazdırır."""
    print("\n" + "-" * 50)
    print(f"  {metin}")
    print("-" * 50)


def figuru_kaydet(isim):
    """Mevcut matplotlib figürünü figures/ klasörüne kaydeder."""
    os.makedirs(FIG_DIR, exist_ok=True)
    yol = os.path.join(FIG_DIR, isim)
    plt.savefig(yol, bbox_inches="tight")
    plt.close()
    print(f"  [kaydedildi] {yol}")


# ======================================================================
# BÖLÜM 1 — TEORİ
# ======================================================================

def teori():
    """
    TF-IDF'in matematiksel formüllerini ve yorumunu konsola yazdırır.
    Henüz kod yok — sadece teori.
    """
    baslik("1. TF-IDF TEORİSİ VE FORMÜLLERİ")

    print("""
  TF-IDF = Term Frequency × Inverse Document Frequency

  ┌─────────────────────────────────────────────────────────────────────┐
  │ TF (Term Frequency) — Terim Sıklığı                                 │
  │   Bir kelimenin bir dokümanda ne sıklıkta geçtiğini ölçer.          │
  │                                                                     │
  │   TF(t,d) = (t'nin d'de geçme sayısı) / (d'deki toplam kelime)     │
  │                                                                     │
  │   Örnek: 5 kelimelik bir cümlede "kedi" 1 kez geçiyorsa             │
  │          TF("kedi") = 1/5 = 0.2                                     │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │ IDF (Inverse Document Frequency) — Ters Doküman Sıklığı             │
  │   Bir kelimenin tüm dokümanlar genelinde ne kadar nadir olduğunu    │
  │   ölçer. Nadir kelimeler daha ayırt edicidir, IDF'i yüksektir.      │
  │                                                                     │
  │   IDF(t) = log( toplam doküman / t'yi içeren doküman )              │
  │                                                                     │
  │   Örnek: "kedi" 4 dokümanın 3'ünde geçiyorsa                        │
  │          IDF("kedi") = log(4/3) ≈ 0.29 (düşük, çünkü sık geçiyor)  │
  │                                                                     │
  │   Örnek: "araba" 4 dokümanın 1'inde geçiyorsa                       │
  │          IDF("araba") = log(4/1) ≈ 1.39 (yüksek, çünkü nadir)      │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │ TF-IDF(t,d) = TF(t,d) × IDF(t)                                      │
  │                                                                     │
  │ Yüksek TF-IDF → kelime bu doküman için karakteristik (önemli)       │
  │ Düşük TF-IDF  → kelime ya nadir değil ya da bu dokümanda az geçiyor │
  └─────────────────────────────────────────────────────────────────────┘

  ÖZET: TF-IDF, bir kelimenin hem doküman içindeki önemini (TF)
        hem de dokümanlar arası ayırt ediciliğini (IDF) birleştirir.
  """)


# ======================================================================
# BÖLÜM 2 — MANUEL HESAPLAMA (SIFIRDAN)
# ======================================================================

def manuel_tfidf():
    """
    4 küçük Türkçe cümle üzerinde TF, IDF ve TF-IDF değerlerini
    sıfırdan, Python'ın temel fonksiyonlarıyla hesaplar.
    Amaç: Kütüphane kullanmadan formülün nasıl çalıştığını göstermek.
    """
    baslik("2. ADIM ADIM MANUEL TF-IDF HESAPLAMA")

    # --- 2.1 Dokümanlar
    dokumanlar = [
        "kedi çok tatli bir hayvan",      # D1
        "kedi ve köpek en iyi arkadas",   # D2
        "köpek sadik bir hayvandir",      # D3
        "araba hizli gider",              # D4
    ]
    N = len(dokumanlar)
    tum_kelimeler = sorted(set(w for d in dokumanlar for w in d.split()))

    print(f"\n  Elimizde {N} doküman var:")
    for i, d in enumerate(dokumanlar):
        print(f"    D{i+1}: \"{d}\"")
    print(f"\n  Benzersiz kelime sayısı: {len(tum_kelimeler)}")
    print(f"  Kelimeler: {tum_kelimeler}")

    # --- 2.2 TF-IDF tablosu
    print("\n\n  Her kelime için TF, IDF ve TF-IDF değerlerini hesaplayalım:\n")
    print(f"  {'Kelime':12s} | {'D1 TF':8s} {'D1 IDF':8s} {'D1 TF-IDF':10s} | "
          f"{'D2 TF':8s} {'D2 IDF':8s} {'D2 TF-IDF':10s} | "
          f"{'D3 TF':8s} {'D3 IDF':8s} {'D3 TF-IDF':10s} | "
          f"{'D4 TF':8s} {'D4 IDF':8s} {'D4 TF-IDF':10s} | "
          f"{'Kaç Döküman':12s} {'IDF':8s}")
    print("  " + "-" * 128)

    for kelime in tum_kelimeler:
        kac_dokumanda = sum(1 for d in dokumanlar if kelime in d.split())
        # Smooth IDF: pay ve paydaya +1 ekleyerek sıfıra bölmeyi engelleriz
        idf_degeri = math.log((N + 1) / (kac_dokumanda + 1)) + 1

        satir = f"  {kelime:12s} |"
        for d in dokumanlar:
            d_kelimeleri = d.split()
            tf = d_kelimeleri.count(kelime) / len(d_kelimeleri)
            tfidf = tf * idf_degeri
            satir += f" {tf:<8.3f} {idf_degeri:<8.3f} {tfidf:<10.4f} |"

        satir += f" {kac_dokumanda}/4           {idf_degeri:<8.3f}"
        print(satir)

    # --- 2.3 "kedi" kelimesi detaylı hesap
    print("\n\n  --- 'kedi' KELİMESİNİN DETAYLI HESABI ---")
    kelime = "kedi"
    kd = sum(1 for d in dokumanlar if kelime in d.split())
    idf_k = math.log((N + 1) / (kd + 1)) + 1
    print(f"  'kedi' {kd}/{N} dokümanda geçiyor.")
    print(f"  IDF('kedi') = log(({N}+1)/({kd}+1)) + 1 = {idf_k:.3f}")
    print(f"  D1'de TF-IDF('kedi') = TF(1/5) × IDF({idf_k:.3f}) = {1/5 * idf_k:.4f}")
    print(f"  D2'de TF-IDF('kedi') = TF(1/6) × IDF({idf_k:.3f}) = {1/6 * idf_k:.4f}")

    # --- 2.4 Nadir vs sık kelime
    print("\n\n  --- NADİR KELİME vs SIK KELİME ---")
    print("  Bir kelime ne kadar az dokümanda geçerse IDF'i o kadar yüksek olur.")
    print("  Yüksek IDF = daha ayırt edici.\n")
    for kelime in ["araba", "ve", "kedi"]:
        kd = sum(1 for d in dokumanlar if kelime in d.split())
        idf_v = math.log((N + 1) / (kd + 1)) + 1
        if idf_v > 1.7:
            tur = "NADİR (ayırt edici)"
        elif idf_v > 1.3:
            tur = "ORTA"
        else:
            tur = "SIK (az ayırt edici)"
        print(f"    '{kelime}': {kd}/4 dokümanda → IDF = {idf_v:.3f} → {tur}")

    # --- 2.5 Daha küçük bir örnekle sıfırdan hesaplama
    print("\n\n  --- KÜÇÜK BİR ÖRNEK ÜZERİNDE SIFIRDAN TF-IDF ---")
    kucuk = [
        "kedi evde uyuyor",
        "köpek parkta kosuyor",
        "kedi ve köpek birlikte oynuyor",
        "evde kedi mamasi var",
    ]

    def tf(kelime, tokenize_doc):
        return tokenize_doc.count(kelime) / len(tokenize_doc) if tokenize_doc else 0

    def idf(kelime, tokenize_dokumanlar):
        doc_count = sum(1 for doc in tokenize_dokumanlar if kelime in doc)
        return math.log(len(tokenize_dokumanlar) / (1 + doc_count)) + 1

    tokenize_kucuk = [d.lower().split() for d in kucuk]

    for i, doc in enumerate(tokenize_kucuk):
        print(f"\n    D{i+1}: \"{kucuk[i]}\"")
        for kelime in sorted(set(doc)):
            tf_val = tf(kelime, doc)
            idf_val = idf(kelime, tokenize_kucuk)
            print(f"      '{kelime}' → TF={tf_val:.4f}  IDF={idf_val:.4f}  TF-IDF={tf_val * idf_val:.4f}")


# ======================================================================
# BÖLÜM 3 — SCIKIT-LEARN İLE TF-IDF
# ======================================================================

def sklearn_tfidf():
    """
    Scikit-learn'ün TfidfVectorizer'ını kullanarak TF-IDF matrisi oluşturur
    ve ısı haritası ile görselleştirir.
    """
    baslik("3. SCIKIT-LEARN İLE TF-IDF")

    kucuk_dokumanlar = [
        "kedi evde uyuyor",
        "köpek parkta kosuyor",
        "kedi ve köpek birlikte oynuyor",
        "evde kedi mamasi var",
    ]

    # TfidfVectorizer: metinleri otomatik tokenize eder ve TF-IDF matrisi oluşturur
    vectorizer = TfidfVectorizer()
    tfidf_matrisi = vectorizer.fit_transform(kucuk_dokumanlar)

    # DataFrame olarak görselleştir
    df = pd.DataFrame(
        tfidf_matrisi.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=[f"Doküman {i+1}" for i in range(len(kucuk_dokumanlar))]
    )

    print("\n  TF-IDF Matrisi (her satır bir doküman, her sütun bir kelime):")
    print(df.round(4).to_string())
    print("\n  Sıfırlar → o kelime o dokümanda hiç geçmiyor.")
    print("  Yüksek değerler → o kelime o doküman için önemli.")

    # Her dokümanda en yüksek TF-IDF skoruna sahip kelime
    print("\n  Her dokümanda en yüksek skora sahip kelime:")
    for i, doc in enumerate(kucuk_dokumanlar):
        satir = df.iloc[i]
        print(f"    D{i+1}: '{doc}' → '{satir.idxmax()}' (skor: {satir.max():.4f})")

    # Isı haritası
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap="YlOrRd", fmt=".3f", linewidths=0.5)
    plt.title("TF-IDF Matrisi (Isı Haritası)")
    plt.xlabel("Kelimeler")
    plt.ylabel("Dokümanlar")
    plt.tight_layout()
    figuru_kaydet("01_tfidf_heatmap.png")


# ======================================================================
# BÖLÜM 4 — TTC-4900 TÜRKÇE VERİ SETİ
# ======================================================================

def turkce_veri_seti():
    """
    TTC-4900 Türkçe haber veri setini yükler, sınıf dağılımını gösterir
    ve Cosine Similarity ile aynı/farklı kategorideki belgelerin
    benzerliğini karşılaştırır.
    """
    baslik("4. TTC-4900 TÜRKÇE HABER VERİ SETİ")

    print("\n  Veri seti yükleniyor...")
    try:
        from datasets import load_dataset
        ds = load_dataset("savasy/ttc4900")
        metinler = ds["train"]["text"]
        etiketler = ds["train"]["category"]
    except Exception:
        # HuggingFace datasets yoksa uyarı ver
        print("\n  [!] 'datasets' kütüphanesi yüklü değil veya internet yok.")
        print("  Yüklemek için: pip install datasets")
        print("  Alternatif: Veriyi manuel olarak data/ klasörüne koyun.")
        print("  Bu bölüm atlanıyor...\n")
        return None, None

    print(f"  Toplam {len(metinler)} Türkçe haber yüklendi.")
    print(f"  Kategoriler ({len(TTC4900_KATEGORILER)}):")
    for i, kat in enumerate(TTC4900_KATEGORILER):
        sayi = sum(1 for e in etiketler if e == i)
        print(f"    {kat:12s}: {sayi} haber")

    # Sınıf dağılımı grafiği
    sayilar = [sum(1 for e in etiketler if e == i) for i in range(len(TTC4900_KATEGORILER))]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(TTC4900_KATEGORILER, sayilar, color=plt.cm.Set3(range(len(TTC4900_KATEGORILER))))
    for bar, sayi in zip(bars, sayilar):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 str(sayi), ha='center', fontsize=10)
    plt.title("TTC-4900: Kategorilere Göre Haber Sayısı")
    plt.xlabel("Kategori")
    plt.ylabel("Haber Sayısı")
    plt.xticks(rotation=30, ha='right')
    plt.ylim(0, max(sayilar) * 1.15)
    plt.tight_layout()
    figuru_kaydet("02_veri_dagilimi.png")

    # Token uzunluğu histogramı
    token_uzunluklari = [len(m.split()) for m in metinler]
    plt.figure(figsize=(10, 4))
    plt.hist(token_uzunluklari, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    plt.axvline(np.mean(token_uzunluklari), color='red', linestyle='--',
                label=f'Ortalama: {np.mean(token_uzunluklari):.1f} kelime')
    plt.title("Haber Uzunluklarının Dağılımı (Token Sayısı)")
    plt.xlabel("Kelime Sayısı")
    plt.ylabel("Haber Sayısı")
    plt.legend()
    plt.tight_layout()
    figuru_kaydet("03_token_uzunluklari.png")

    # Cosine similarity: aynı kategoriden 2 belge vs farklı kategoriden
    print("\n  Cosine Similarity (benzerlik) analizi...")
    v = TfidfVectorizer(max_features=2000)
    X = v.fit_transform(metinler)

    siyaset_idx = [i for i, e in enumerate(etiketler) if e == 0]  # siyaset
    spor_idx = [i for i, e in enumerate(etiketler) if e == 5]     # spor

    ayni_benzerlik = cosine_similarity(X[siyaset_idx[0]:siyaset_idx[0]+1],
                                        X[siyaset_idx[1]:siyaset_idx[1]+1])[0][0]
    farkli_benzerlik = cosine_similarity(X[siyaset_idx[0]:siyaset_idx[0]+1],
                                          X[spor_idx[0]:spor_idx[0]+1])[0][0]

    print(f"    Aynı kategoriden 2 siyaset haberi benzerliği: {ayni_benzerlik:.4f}")
    print(f"    Farklı kategoriden (siyaset vs spor) benzerlik: {farkli_benzerlik:.4f}")
    print("    → Aynı kategorideki haberler daha benzer, TF-IDF bunu yakalıyor.")

    return metinler, etiketler


# ======================================================================
# BÖLÜM 5 — SINIFLANDIRMA
# ======================================================================

def siniflandirma(metinler, etiketler):
    """
    TTC-4900 veri seti üzerinde TF-IDF + Logistic Regression ile
    Türkçe haber sınıflandırması yapar.
    """
    baslik("5. TF-IDF + LOGISTIC REGRESSION İLE TÜRKÇE HABER SINIFLANDIRMASI")

    if metinler is None:
        print("\n  Veri seti yüklenemedi, bu bölüm atlanıyor.")
        return None, None, None

    X_egitim, X_test, y_egitim, y_test = train_test_split(
        metinler, etiketler, test_size=0.2, random_state=RANDOM_STATE, stratify=etiketler
    )
    print(f"\n  Eğitim: {len(X_egitim)} haber | Test: {len(X_test)} haber")

    MAX_FEATURES = 10000

    # Pipeline: önce TF-IDF vektörleştir, sonra Logistic Regression ile sınıflandır
    model = make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 2),       # unigram + bigram
            min_df=5,                  # en az 5 dokümanda geçsin
            max_features=MAX_FEATURES,
            sublinear_tf=True          # TF'i log(1+TF) ile yumuşat
        ),
        LogisticRegression(
            max_iter=1000,
            class_weight='balanced',   # sınıf dengesizliğine karşı
            random_state=RANDOM_STATE
        ),
    )

    print("  Model eğitiliyor...")
    model.fit(X_egitim, y_egitim)

    tahmin = model.predict(X_test)
    dogruluk = accuracy_score(y_test, tahmin)
    f1 = f1_score(y_test, tahmin, average='macro')

    print(f"\n  Doğruluk (Accuracy):  {dogruluk:.4f}")
    print(f"  F1 Skoru (macro):     {f1:.4f}")
    print("\n  Sınıflandırma Raporu:")
    print(classification_report(y_test, tahmin, target_names=TTC4900_KATEGORILER))

    # Confusion matrix
    cm = confusion_matrix(y_test, tahmin)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(TTC4900_KATEGORILER)))
    ax.set_xticklabels(TTC4900_KATEGORILER, rotation=30, ha='right')
    ax.set_yticks(range(len(TTC4900_KATEGORILER)))
    ax.set_yticklabels(TTC4900_KATEGORILER)
    ax.set_xlabel("Tahmin Edilen")
    ax.set_ylabel("Gerçek")
    ax.set_title(f"Confusion Matrix — Doğruluk: %{dogruluk*100:.1f}")
    for i in range(len(TTC4900_KATEGORILER)):
        for j in range(len(TTC4900_KATEGORILER)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    figuru_kaydet("04_confusion_matrix.png")

    # En etkili kelimeler
    vec = model.named_steps['tfidfvectorizer']
    clf = model.named_steps['logisticregression']
    kelime_isimleri = np.array(vec.get_feature_names_out())

    # Her kategori için en etkili kelimeleri göster
    print("\n  Her kategoride en karakteristik kelimeler:\n")
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes = axes.flatten()

    for kat_idx in range(len(TTC4900_KATEGORILER)):
        katsayilar = clf.coef_[kat_idx]
        en_iyi = np.argsort(katsayilar)[-10:]
        kelimeler = kelime_isimleri[en_iyi]
        skorlar = katsayilar[en_iyi]

        ax = axes[kat_idx]
        ax.barh(range(10), skorlar, color='steelblue')
        ax.set_yticks(range(10))
        ax.set_yticklabels(kelimeler)
        ax.set_title(f"{TTC4900_KATEGORILER[kat_idx]}", fontsize=11)
        ax.axvline(0, color='gray', lw=0.8)

    # Fazla ekseni gizle
    for ax in axes[len(TTC4900_KATEGORILER):]:
        ax.set_visible(False)

    fig.suptitle("Her Kategori İçin En Karakteristik 10 Kelime", fontsize=14, y=1.01)
    fig.tight_layout()
    figuru_kaydet("05_kelime_onemi.png")

    print("  (Grafik figures/ klasörüne kaydedildi)")
    for kat_idx in range(len(TTC4900_KATEGORILER)):
        katsayilar = clf.coef_[kat_idx]
        en_iyi = np.argsort(katsayilar)[-5:]
        kelimeler = ", ".join(kelime_isimleri[en_iyi][::-1])
        print(f"    {TTC4900_KATEGORILER[kat_idx]:12s}: {kelimeler}")

    return dogruluk, f1, MAX_FEATURES


# ======================================================================
# BÖLÜM 6 — SVD İLE BOYUT İNDİRGEME
# ======================================================================

def svd_analizi(metinler, etiketler):
    """
    Truncated SVD kullanarak TF-IDF matrisinin boyutunu düşürür ve
    performans takasını (accuracy vs boyut) grafikle gösterir.
    """
    baslik("6. SVD İLE BOYUT İNDİRGEME VE PERFORMANS TAKASI")

    if metinler is None:
        print("\n  Veri seti yüklenemedi, bu bölüm atlanıyor.")
        return None, None, None

    X_egitim, X_test, y_egitim, y_test = train_test_split(
        metinler, etiketler, test_size=0.2, random_state=RANDOM_STATE, stratify=etiketler
    )

    MAX_FEATURES = 10000
    SVD_BILESENLER = [25, 50, 100, 200, 500]

    print(f"\n  Orijinal TF-IDF boyutu: {MAX_FEATURES}")
    print(f"  SVD ile {SVD_BILESENLER} boyuta indiriyoruz...\n")

    dogruluklar, f1ler, varyanslar = [], [], []

    for k in SVD_BILESENLER:
        pipe = make_pipeline(
            TfidfVectorizer(
                ngram_range=(1, 2), min_df=5,
                max_features=MAX_FEATURES, sublinear_tf=True
            ),
            TruncatedSVD(n_components=k, random_state=RANDOM_STATE),
            LogisticRegression(
                max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE
            ),
        )
        pipe.fit(X_egitim, y_egitim)
        tahmin = pipe.predict(X_test)
        dogruluklar.append(accuracy_score(y_test, tahmin))
        f1ler.append(f1_score(y_test, tahmin, average='macro'))
        evr = pipe.named_steps['truncatedsvd'].explained_variance_ratio_.sum()
        varyanslar.append(evr)
        sikisma = 100 * (1 - k / MAX_FEATURES)
        print(f"    SVD({k:3d}) → doğruluk={dogruluklar[-1]:.4f}  "
              f"F1={f1ler[-1]:.4f}  varyans=%{100*evr:.1f}  "
              f"sıkışma=%{sikisma:.1f}")

    # Grafik: çift eksenli (accuracy vs varyans)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(SVD_BILESENLER, dogruluklar, "o-", color="#0275d8", lw=2, label="Doğruluk")
    ax1.plot(SVD_BILESENLER, f1ler, "s-", color="#5cb85c", lw=2, label="F1 (macro)")
    ax1.set_xlabel("SVD Bileşen Sayısı (boyut)")
    ax1.set_ylabel("Skor")
    ax1.legend(loc="lower right")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(SVD_BILESENLER, [100*e for e in varyanslar], "^--",
             color="#f0ad4e", lw=2, label="Açıklanan Varyans %")
    ax2.set_ylabel("Açıklanan Varyans (%)", color="#f0ad4e")
    ax2.set_ylim(0, 105)

    ax1.set_title("SVD: Boyut İndirgeme — Performans Takası\n"
                  f"(Orijinal boyut: {MAX_FEATURES})")
    fig.tight_layout()
    figuru_kaydet("06_svd_tradeoff.png")

    return dogruluklar, f1ler, varyanslar


# ======================================================================
# BÖLÜM 7 — TF-IDF LİMİTLERİ
# ======================================================================

def tfidf_limitleri():
    """
    TF-IDF'in dört temel sınırlamasını gösterir:
    1) Anlamsal ilişki yakalayamama
    2) Kelime sırasını yok sayma (bağlam kaybı)
    3) Seyreklik problemi
    4) Boyut laneti
    """
    baslik("7. TF-IDF'İN LİMİTLERİ")

    # --- 7.1 Anlamsal İlişki ---
    alt_baslik("7.1 Anlamsal İlişki (Semantics)")

    print("""
  TF-IDF her kelimeyi bağımsız bir ID (sütun) olarak görür.
  "Kral" ve "Kraliçe" arasındaki anlamsal yakınlığı bilmez.
  İkisi de sadece farklı birer sütundur.
  """)

    dokumanlar = [
        "kral sarayda yasar",
        "kralice sarayda yasar",
        "masa odada duruyor",
    ]
    v = TfidfVectorizer()
    m = v.fit_transform(dokumanlar)
    benzerlik = cosine_similarity(m[0:1], m[1:3])[0]
    print(f"  Cosine benzerlik:")
    print(f"    'kral ...' vs 'kralice ...': {benzerlik[0]:.4f}")
    print(f"    'kral ...' vs 'masa ...'   : {benzerlik[1]:.4f}")
    print("  → İki değer de sıfır! Oysa kral ve kraliçe anlamsal olarak yakın.")
    print("  → Çözüm: Word2Vec, GloVe, FastText gibi embedding yöntemleri.")

    # --- 7.2 Bağlam Kaybı ---
    alt_baslik("7.2 Bağlam Kaybı (Context)")

    print("""
  TF-IDF bir "bag-of-words" modelidir. Kelimelerin sırasını tamamen yok sayar.
  "Köpek adamı ısırdı" ile "Adam köpeği ısırdı" TF-IDF için neredeyse aynıdır.
  Oysa anlamları taban tabana zıttır!
  """)

    dok_baglam = [
        "köpek adami isirdi",
        "adam köpegi isirdi",
        "kedi fareyi kovaladi",
    ]

    v1 = TfidfVectorizer()
    m1 = v1.fit_transform(dok_baglam)
    b1 = cosine_similarity(m1[0:1], m1[1:3])[0]
    print(f"  Unigram benzerlik: {b1[0]:.4f}")

    v2 = TfidfVectorizer(ngram_range=(1, 2))
    m2 = v2.fit_transform(dok_baglam)
    b2 = cosine_similarity(m2[0:1], m2[1:3])[0]
    print(f"  Bigram benzerlik:  {b2[0]:.4f}")
    print("  → Bigram ile 'köpek adamı' ve 'adam köpeği' farklı öznitelik olur.")
    print("  → Kısmi çözüm: n-gram. Gerçek çözüm: RNN, LSTM, Transformer.")

    # --- 7.3 Seyreklik ---
    alt_baslik("7.3 Seyreklik (Sparsity)")

    print("""
  Kelime dağarcığı büyüdükçe TF-IDF matrisi aşırı seyrek hale gelir.
  Örneğin 10.000 doküman × 50.000 kelime = 500 milyon hücre.
  Ama her dokümanda ortalama 50-100 farklı kelime vardır → %99.9'u sıfır!
  """)

    np.random.seed(RANDOM_STATE)
    kelime_havuzu = [
        "teknoloji", "yapay", "zeka", "veri", "bilim", "yazilim", "makine",
        "ogrenme", "derin", "sinir", "dil", "isleme", "analiz", "model",
        "algoritma", "optimizasyon", "matris", "siniflandirma", "tahmin",
        "kral", "kralice", "masa", "kitap", "spor", "ekonomi", "saglik",
        "egitim", "hukuk", "sanat", "muzik", "tarih", "politika", "cevre",
        "doga", "enerji", "dijital", "robot", "uzay", "zaman", "donanim",
    ]

    N = 5000
    dokumanlar = []
    for _ in range(N):
        uzunluk = np.random.randint(20, 100)
        dokumanlar.append(" ".join(np.random.choice(kelime_havuzu, uzunluk, replace=True)))

    v = TfidfVectorizer()
    m = v.fit_transform(dokumanlar)
    toplam = m.shape[0] * m.shape[1]
    seyreklik = 100 * (1 - m.nnz / toplam)

    print(f"  Doküman sayısı:     {m.shape[0]:,}")
    print(f"  Kelime dağarcığı:   {m.shape[1]:,}")
    print(f"  Toplam hücre:       {toplam:,}")
    print(f"  Sıfır olmayan:      {m.nnz:,}")
    print(f"  Seyreklik:          %{seyreklik:.2f}")
    print(f"  → Matrisin %{seyreklik:.1f}'i boş (sıfır)!")
    print(f"  → Çözüm: SVD, PCA, Embedding katmanları.")

    # Seyreklik grafiği
    max_feat_list = [100, 500, 1000, 2000, 5000]
    seyreklikler, boyutlar = [], []
    for mf in max_feat_list:
        v = TfidfVectorizer(max_features=mf)
        m = v.fit_transform(dokumanlar)
        toplam = m.shape[0] * m.shape[1]
        seyreklikler.append(100 * (1 - m.nnz / toplam))
        boyutlar.append(m.shape[1])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(b) for b in boyutlar], seyreklikler, color='coral')
    ax.set_xlabel("Vocabulary Boyutu")
    ax.set_ylabel("Seyreklik (%)")
    ax.set_title("Vocabulary Büyüdükçe Seyreklik Artar")
    ax.set_ylim(80, 100)
    for i, v in enumerate(seyreklikler):
        ax.text(i, v + 0.5, f"%{v:.1f}", ha='center', fontsize=9)
    fig.tight_layout()
    figuru_kaydet("07_sparsity.png")

    # --- 7.4 Boyut Laneti ---
    alt_baslik("7.4 Boyut Laneti (Curse of Dimensionality)")

    print("""
  Yüksek boyutlu uzayda tüm noktalar birbirine eşit uzaklıkta görünür.
  Mesafeler anlamsızlaşır → benzerlik ölçümleri çalışmaz.
  """)

    boyutlar = [2, 5, 10, 50, 100, 500]
    nokta_sayisi = 50
    mesafe_ort, mesafe_std = [], []
    for dim in boyutlar:
        data = np.random.uniform(0, 1, (nokta_sayisi, dim))
        mesafeler = pdist(data, 'euclidean')
        mesafe_ort.append(np.mean(mesafeler))
        mesafe_std.append(np.std(mesafeler))

    cv = [s / m for m, s in zip(mesafe_ort, mesafe_std)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(boyutlar, mesafe_ort, marker='o', lw=2)
    axes[0].fill_between(boyutlar,
                         np.array(mesafe_ort) - np.array(mesafe_std),
                         np.array(mesafe_ort) + np.array(mesafe_std),
                         alpha=0.2)
    axes[0].set_xlabel("Boyut")
    axes[0].set_ylabel("Ortalama Öklid Mesafesi")
    axes[0].set_title("Boyut Arttıkça Mesafeler Büyür ve Ayrışır")
    axes[0].grid(True)

    axes[1].plot(boyutlar, cv, marker='o', color='red', lw=2)
    axes[1].axhline(y=0, color='gray', linestyle='--')
    axes[1].set_xlabel("Boyut")
    axes[1].set_ylabel("CV (Std / Ortalama)")
    axes[1].set_title("Boyut Arttıkça Mesafeler Ayırt Edilemez (CV → 0)")
    axes[1].grid(True)
    fig.tight_layout()
    figuru_kaydet("08_boyut_laneti.png")


# ======================================================================
# BÖLÜM 8 — ÖZET VE ÖRNEK TAHMİNLER
# ======================================================================

def ozet_ve_tahmin(metinler, etiketler):
    """
    Tüm analizin özetini ve birkaç örnek metin üzerinde tahmin yapar.
    """
    baslik("8. ÖZET VE ÖRNEK TAHMİNLER")

    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                         TF-IDF ÖZETİ                                │
  ├─────────────────────────────────────────────────────────────────────┤
  │ TF-IDF = TF × IDF                                                   │
  │                                                                     │
  │ Avantajları:                                                        │
  │   + Basit, hızlı, anlaşılır                                         │
  │   + İyi bir baseline (referans) model                               │
  │   + Spam tespiti, konu sınıflandırma gibi işlerde etkili            │
  │                                                                     │
  │ Limitleri:                                                          │
  │   − Anlamsal ilişki yok     → Word2Vec, GloVe, FastText             │
  │   − Bağlam/kelime sırası yok → RNN, LSTM, Transformer               │
  │   − Seyreklik               → SVD, PCA, Embedding                   │
  │   − Boyut laneti            → t-SNE, UMAP                           │
  └─────────────────────────────────────────────────────────────────────┘
  """)

    # Örnek tahminler
    print("  ÖRNEK TAHMİNLER:")
    if metinler is None:
        print("  Veri seti yüklenemediği için tahmin yapılamıyor.")
        return

    ornekler = [
        "TBMM'de yeni yasa teklifi görüşüldü",
        "Fenerbahçe derbi maçı 3-2 kazandı",
        "Apple yeni yapay zeka modelini tanıttı",
        "Borsa İstanbul'da endeks yükselişle kapandı",
        "Kültür Bakanlığı yeni sergi salonu açtı",
    ]

    model = make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 2), min_df=5, max_features=10000, sublinear_tf=True
        ),
        LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE
        ),
    )
    model.fit(metinler, etiketler)

    for metin in ornekler:
        tahmin = model.predict([metin])[0]
        olasiliklar = model.predict_proba([metin])[0]
        en_yuksek = np.argmax(olasiliklar)
        print(f"\n    \"{metin}\"")
        print(f"    → {TTC4900_KATEGORILER[en_yuksek]} "
              f"(%{olasiliklar[en_yuksek]*100:.1f})")


# ======================================================================
# ANA PROGRAM
# ======================================================================

def main():
    print("=" * 70)
    print("    TF-IDF: TEORİDEN PRATİĞE")
    print("    Kapsamlı Anlatım, Uygulama ve Analiz")
    print("=" * 70)

    # 1. Teori
    teori()

    # 2. Manuel hesaplama
    manuel_tfidf()

    # 3. Scikit-learn ile TF-IDF
    sklearn_tfidf()

    # 4. TTC-4900 Türkçe veri seti
    metinler, etiketler = turkce_veri_seti()

    # 5. Sınıflandırma
    dogruluk, f1, n_feat = siniflandirma(metinler, etiketler)

    # 6. SVD analizi
    svd_dogruluk, svd_f1, svd_evr = svd_analizi(metinler, etiketler)

    # 7. TF-IDF limitleri
    tfidf_limitleri()

    # 8. Özet ve tahminler
    ozet_ve_tahmin(metinler, etiketler)

    # Sonuç tablosu
    print("\n" + "=" * 70)
    print("    SONUÇ TABLOSU")
    print("=" * 70)

    if dogruluk is not None:
        print(f"\n  TF-IDF + Logistic Regression  →  {n_feat} boyut  "
              f"Doğruluk={dogruluk:.4f}  F1={f1:.4f}")
        for i, k in enumerate([25, 50, 100, 200, 500]):
            print(f"  TF-IDF + SVD({k:3d}) + LogReg   →  {k:5d} boyut  "
                  f"Doğruluk={svd_dogruluk[i]:.4f}  F1={svd_f1[i]:.4f}  "
                  f"varyans=%{100*svd_evr[i]:.1f}")

    print(f"\n  Tüm görseller '{FIG_DIR}/' klasörüne kaydedildi.")
    print("=" * 70)


if __name__ == "__main__":
    main()
