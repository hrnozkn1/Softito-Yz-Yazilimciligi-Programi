#!/usr/bin/env python3
"""
rnn_haber_siniflandirma.py
===========================
Türkçe Haber Sınıflandırması — Vanilla RNN (PyTorch)

RNN (Recurrent Neural Network), sıralı verilerle (metin, zaman serisi)
çalışmak için tasarlanmış bir sinir ağı mimarisidir. TF-IDF'ten farklı
olarak kelimelerin sırasını dikkate alır.

Veri seti : TTC-4900 (Türkçe Haber Kategorizasyonu, 7 kategori)
Model     : Embedding → nn.RNN(tanh) → Dropout → Linear(7)

Çalıştırma:
  pip install -r requirements.txt
  python rnn_haber_siniflandirma.py

Üretilenler:
  figures/ -> EDA, eğitim eğrileri, confusion matrix (PNG)
"""

import os, re, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ======================================================================
# AYARLAR
# ======================================================================
SEED = 42
DATA_DIR = "data"
FIGURES_DIR = "figures"
BATCH_SIZE = 32
EPOCHS = 20
LR = 2e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 1
MAX_SEQ_LEN = 50
TEST_SIZE = 0.2
EMBED_DIM = 128

KATEGORILER = {
    0: "siyaset", 1: "ekonomi", 2: "kultur",
    3: "saglik", 4: "spor", 5: "teknoloji", 6: "dunya"
}
NUM_CLASSES = len(KATEGORILER)

os.makedirs(FIGURES_DIR, exist_ok=True)


def baslik(metin):
    print("\n" + "=" * 60)
    print(f"  {metin}")
    print("=" * 60)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Cihaz: {DEVICE}")


# ======================================================================
# 1 — VERİ YÜKLEME
# ======================================================================

def load_data():
    """TTC-4900 verisini CSV'den yükler, yoksa sentetik fallback oluşturur."""
    csv_path = os.path.join(DATA_DIR, "ttc4900.csv")

    if not os.path.exists(csv_path):
        print(f"\n  [!] Veri dosyası bulunamadı: {csv_path}")
        print(f"  İndirmek için Kaggle: savasy/ttc4900")
        print(f"  Veya HuggingFace: datasets.load_dataset('savasy/ttc4900')")
        return create_fallback_data()

    df = pd.read_csv(csv_path)
    print(f"  Veri yüklendi: {len(df)} haber, {df['category'].nunique()} kategori")
    return df


def create_fallback_data():
    """Sentetik Türkçe haber verisi oluşturur (her kategoriden 700 örnek)."""
    np.random.seed(SEED)
    ornekler = {
        0: ["hükümet yeni yasa teklifini meclise sundu muhalefet sert tepki gösterdi",
            "cumhurbaşkanı başkanlık sistemi ile ilgili açıklamalarda bulundu",
            "seçim tarihi ile ilgili yeni gelişmeler yaşanıyor",
            "milletvekili aday tanıtım toplantısı yapıldı",
            "partiler koalisyon görüşmelerine devam ediyor"],
        1: ["merkez bankası faiz oranlarını sabit tuttu",
            "borsa günü yükselişle kapattı yatırımcılar raporları bekliyor",
            "enflasyon verileri açıklandı piyasalarda dalgalanma var",
            "ekonomi bakanı yeni teşvik paketini duyurdu",
            "dolar kuru bugün 30 lira seviyesinde işlem görüyor"],
        2: ["tiyatro festivali bu yıl 25 sanatçıyı ağırlayacak",
            "kitap fuarı ziyaretçilere kapılarını açtı",
            "ressamın eserleri sergide sanatseverlerle buluşuyor",
            "konser etkinliğinde ünlü müzisyenler sahne aldı",
            "kültür merkezinde halk oyunları gösterisi düzenlendi"],
        3: ["sağlık bakanı yeni hastane projelerini tanıttı",
            "koronavirüs vaka sayılarında azalma görülüyor",
            "doktorlar grip aşısı konusunda uyarılarda bulundu",
            "sağlıklı beslenme ve diyet önerileri uzmanlardan geldi",
            "sigaranın zararları ile ilgili kampanya başlatıldı"],
        4: ["futbol takımı maçı son dakika golüyle kazandı",
            "milli sporcu olimpiyatlarda altın madalya kazandı",
            "basketbol liginde şampiyon belli oldu",
            "tenis turnuvasında sürpriz sonuçlar yaşandı",
            "voleybol takımı rakiplerini yenerek finale yükseldi"],
        5: ["yapay zeka teknolojileri sağlık sektöründe kullanılmaya başlandı",
            "sosyal medya platformu yeni özellikler duyurdu",
            "akıllı telefon pazarında rekor satış rakamları açıklandı",
            "bulut bilişim hizmetleri şirketler tarafından yoğun ilgi görüyor",
            "siber güvenlik uzmanları yeni tehditlere karşı uyarıyor"],
        6: ["birleşmiş milletler iklim değişikliği raporunu yayınladı",
            "abd ve çin arasındaki ticaret görüşmeleri devam ediyor",
            "avrupa birliği yeni düzenlemeleri onayladı",
            "ortadoğu'da barış görüşmeleri yeniden başladı",
            "afrika kıtasında ekonomik büyüme hızlandı"]
    }
    rows = []
    for cat_id, texts in ornekler.items():
        for _ in range(140):
            for t in texts:
                rows.append({"category": cat_id, "text": t})
    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df = df.groupby("category").head(700).reset_index(drop=True)
    print(f"  Fallback veri: {len(df)} haber oluşturuldu.")
    return df


# ======================================================================
# 2 — TOKENİZASYON VE SÖZLÜK
# ======================================================================

def turkce_tokenize(text):
    """Türkçe metni küçük harf yapar, noktalama işaretlerini siler, token listesi döndürür."""
    text = str(text).lower()
    text = re.sub(r'[^a-zöçşığüâîû0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()


def sozluk_olustur(token_listesi, min_freq=5):
    """Token listelerinden sözlük (word → id) oluşturur."""
    frekans = {}
    for tokens in token_listesi:
        for t in tokens:
            frekans[t] = frekans.get(t, 0) + 1

    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for w, c in sorted(frekans.items()):
        if c >= min_freq:
            vocab[w] = idx
            idx += 1
    return vocab


def kodla(tokens, vocab, max_len):
    """Token listesini ID dizisine çevirir, padding/kesme uygular."""
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids


# ======================================================================
# 3 — VERİ HAZIRLAMA
# ======================================================================

def veri_hazirla(df):
    """Tokenize, sözlük oluştur, kodla, train/test böl."""
    baslik("Veri Hazırlanıyor")

    print("  Tokenize ediliyor...")
    df["tokens"] = df["text"].apply(turkce_tokenize)

    print("  Sözlük oluşturuluyor...")
    vocab = sozluk_olustur(df["tokens"].tolist())
    vocab_size = len(vocab)
    print(f"  Sözlük boyutu: {vocab_size}")

    print("  Diziler kodlanıyor...")
    df["encoded"] = df["tokens"].apply(lambda x: kodla(x, vocab, MAX_SEQ_LEN))
    df["seq_len"] = df["tokens"].apply(len)
    print(f"  Ortalama token: {df['seq_len'].mean():.1f} | Max: {df['seq_len'].max()}")

    # Train/Test bölme
    X = np.array(df["encoded"].tolist())
    y = df["category"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"  Eğitim: {len(X_tr)} | Test: {len(X_te)}")

    return df, vocab, vocab_size, X_tr, X_te, y_tr, y_te


# ======================================================================
# 4 — EDA
# ======================================================================

def eda_grafikler(df):
    """Sınıf dağılımı, token uzunluğu ve en sık kelimeler grafiği."""
    baslik("Keşifsel Veri Analizi (EDA)")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("TTC-4900 Türkçe Haber Veri Seti", fontsize=14, fontweight="bold")

    cat_names = [KATEGORILER[i] for i in range(NUM_CLASSES)]
    counts = df["category"].value_counts().sort_index()
    axes[0].bar(cat_names, counts.values, color=sns.color_palette("Set2", NUM_CLASSES))
    axes[0].set_title("Sınıf Dağılımı")
    axes[0].set_ylabel("Haber Sayısı")
    axes[0].tick_params(axis="x", rotation=45)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontsize=9)

    axes[1].hist(df["seq_len"], bins=40, color="#4C72B0", edgecolor="white")
    axes[1].axvline(MAX_SEQ_LEN, color="red", linestyle="--",
                    label=f"MAX_SEQ_LEN={MAX_SEQ_LEN}")
    axes[1].set_title("Token Sayısı Dağılımı")
    axes[1].set_xlabel("Token Sayısı")
    axes[1].set_ylabel("Haber Sayısı")
    axes[1].legend()

    stop_words = {"bir", "bu", "da", "de", "daha", "en", "ile", "için", "ise",
                  "kadar", "olan", "olarak", "old", "ve", "veya", "ya", "ne",
                  "mi", "çok", "her", "gibi", "ki", "ama", "ancak", "fakat",
                  "çünkü", "eğer", "ayrıca", "bile", "hatta"}
    tum_tokenler = [t for tokens in df["tokens"] for t in tokens]
    frekans = Counter([t for t in tum_tokenler if t not in stop_words and len(t) > 2])
    top = frekans.most_common(15)
    labels_w, vals = zip(*top)
    axes[2].barh(labels_w[::-1], vals[::-1], color="#4C72B0")
    axes[2].set_title("En Sık 15 Kelime (stop words hariç)")
    axes[2].set_xlabel("Frekans")

    plt.tight_layout()
    yol = os.path.join(FIGURES_DIR, "01_eda.png")
    plt.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [kaydedildi] {yol}")


# ======================================================================
# 5 — VERİ SETİ SINIFI
# ======================================================================

class HaberDataset(Dataset):
    def __init__(self, encodings, labels):
        self.X = torch.tensor(np.array(encodings), dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ======================================================================
# 6 — VANILLA RNN MODELİ
# ======================================================================

class VanillaRNN(nn.Module):
    """Embedding → nn.RNN(tanh) → Dropout → Linear sınıflandırıcı."""
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh"
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)       # (batch, seq, embed)
        _, hidden = self.rnn(embedded)     # hidden: (num_layers, batch, hidden)
        out = self.dropout(hidden[-1])     # son katmanın çıktısı
        return self.fc(out)


# ======================================================================
# 7 — EĞİTİM
# ======================================================================

def egit(X_tr, X_te, y_tr, y_te, vocab_size):
    """Modeli eğitir, eğitim eğrilerini ve test sonuçlarını üretir."""
    baslik("VANILLA RNN EĞİTİMİ")

    train_loader = DataLoader(
        HaberDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        HaberDataset(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False
    )

    model = VanillaRNN(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
    model = model.to(DEVICE)
    print(f"  Model: Embedding({vocab_size}→{EMBED_DIM}) + "
          f"RNN({EMBED_DIM}→{HIDDEN_SIZE}) + Linear({HIDDEN_SIZE}→{NUM_CLASSES})")
    print(f"  Parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, val_accs, val_f1s = [], [], []

    for epoch in range(1, EPOCHS + 1):
        # --- Eğitim ---
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"  Epoch {epoch}/{EPOCHS}",
                                     leave=False):
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(y_batch)

        avg_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)

        # --- Doğrulama ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                logits = model(X_batch.to(DEVICE))
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(y_batch.tolist())

        acc = accuracy_score(all_labels, all_preds) * 100
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        val_accs.append(acc)
        val_f1s.append(f1)
        print(f"    Loss: {avg_loss:.4f}  Acc: {acc:.1f}%  F1: {f1:.4f}")

    return model, test_loader, train_losses, val_accs, val_f1s


# ======================================================================
# 8 — GRAFİKLER
# ======================================================================

def egitim_egrileri(train_losses, val_accs, val_f1s, y_te):
    """Eğitim kaybı ve doğruluk/F1 eğrilerini çizer."""
    BG, BLUE, GREEN, ORANGE = "#1e1e2e", "#5c7cfa", "#40c057", "#ff922b"

    with plt.style.context("dark_background"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor(BG)
        for ax in (ax1, ax2):
            ax.set_facecolor(BG)

        ax1.plot(range(EPOCHS), train_losses, color=BLUE, linewidth=2)
        ax1.set_title("Eğitim Kaybı", color="white", fontsize=12)
        ax1.set_xlabel("Epoch", color="white"); ax1.set_ylabel("Kayıp", color="white")
        ax1.tick_params(colors="white"); ax1.grid(color="#3a3a5c", linewidth=0.7)

        baseline = (max(np.bincount(y_te)) / len(y_te)) * 100
        ax2.plot(range(EPOCHS), val_accs, color=GREEN, linewidth=2)
        ax2.plot(range(EPOCHS), val_f1s, color=ORANGE, linewidth=2, linestyle="--")
        ax2.axhline(baseline, color="gray", linestyle=":",
                    label=f"Çoğunluk tahmini ({baseline:.0f}%)")
        ax2.set_title("Doğruluk ve F1 (%)", color="white", fontsize=12)
        ax2.set_xlabel("Epoch", color="white"); ax2.set_ylabel("Skor (%)", color="white")
        ax2.set_ylim(0, 100)
        ax2.tick_params(colors="white"); ax2.grid(color="#3a3a5c", linewidth=0.7)
        ax2.legend(facecolor=BG, edgecolor="#3a3a5c", labelcolor="white")

        plt.tight_layout()
        yol = os.path.join(FIGURES_DIR, "02_training_curves.png")
        plt.savefig(yol, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close()
    print(f"  [kaydedildi] {yol}")


def degerlendir(model, test_loader):
    """Test seti üzerinde nihai değerlendirme yapar."""
    baslik("TEST SONUÇLARI")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch.to(DEVICE))
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y_batch.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_w = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"  Doğruluk:      {acc:.4f}")
    print(f"  F1 (macro):    {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_w:.4f}")
    print(f"\n{classification_report(
        all_labels, all_preds,
        target_names=[KATEGORILER[i] for i in range(NUM_CLASSES)],
        zero_division=0
    )}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[KATEGORILER[i] for i in range(NUM_CLASSES)],
                yticklabels=[KATEGORILER[i] for i in range(NUM_CLASSES)], ax=ax)
    ax.set_title("Confusion Matrix — Vanilla RNN", fontsize=13, fontweight="bold")
    ax.set_ylabel("Gerçek"); ax.set_xlabel("Tahmin")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    yol = os.path.join(FIGURES_DIR, "03_confusion_matrix.png")
    plt.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [kaydedildi] {yol}")


# ======================================================================
# 9 — ÖRNEK TAHMİNLER
# ======================================================================

def ornek_tahminler(model, X_te, y_te):
    """Rastgele 5 test örneği üzerinde tahminleri gösterir."""
    baslik("ÖRNEK TAHMİNLER")

    indices = np.random.choice(len(X_te), 5, replace=False)
    model.eval()
    dogru = 0
    for idx in indices:
        seq = torch.tensor([X_te[idx]], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            logits = model(seq)
            prob = torch.softmax(logits, dim=1)
            pred_id = torch.argmax(logits, dim=1).item()
            guven = prob[0, pred_id].item()

        true_id = y_te[idx]
        isaret = "✓" if pred_id == true_id else "✗"
        if pred_id == true_id:
            dogru += 1
        print(f"    {isaret}  Gerçek: {KATEGORILER[true_id]:10s} → "
              f"Tahmin: {KATEGORILER[pred_id]:10s} (%{guven*100:.0f})")

    print(f"\n  {dogru}/5 doğru")


# ======================================================================
# ANA PROGRAM
# ======================================================================

def main():
    print("=" * 60)
    print("    VANILLA RNN — TÜRKÇE HABER SINIFLANDIRMASI")
    print("    Embedding → nn.RNN(tanh) → Linear")
    print("=" * 60)

    # 1. Veri yükle
    baslik("Veri Yükleniyor")
    df = load_data()

    # 2. Tokenize + sözlük + kodlama + split
    df, vocab, vocab_size, X_tr, X_te, y_tr, y_te = veri_hazirla(df)

    # 3. EDA
    eda_grafikler(df)

    # 4. Eğitim
    model, test_loader, train_losses, val_accs, val_f1s = egit(
        X_tr, X_te, y_tr, y_te, vocab_size
    )

    # 5. Eğitim eğrileri
    egitim_egrileri(train_losses, val_accs, val_f1s, y_te)

    # 6. Nihai değerlendirme
    degerlendir(model, test_loader)

    # 7. Örnek tahminler
    ornek_tahminler(model, X_te, y_te)

    print("\n" + "=" * 60)
    print(f"    TAMAMLANDI — Görseller '{FIGURES_DIR}/' klasöründe.")
    print("=" * 60)


if __name__ == "__main__":
    main()
