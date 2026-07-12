#!/usr/bin/env python3
"""
attention_siniflandirma.py
===========================
AG News Sınıflandırması — LSTM + Attention (PyTorch)

BiLSTM + Bahdanau Attention ile haber sınıflandırması.
Attention mekanizması, modelin dizinin hangi kısımlarına odaklandığını
gösterir — modelin "kararını" yorumlanabilir kılar.

Veri seti : AG News (4 kategorili İngilizce haber, 120K örnek)
Model     : Embedding → BiLSTM → Bahdanau Attention → Linear(4)

Fark: LSTM sadece son hidden state'i kullanırken, Attention TÜM çıktılara
      ağırlıklı olarak bakar. Bu, bilgi kaybını azaltır.

Çalıştırma:
  pip install -r requirements.txt
  python attention_siniflandirma.py

Üretilenler:
  figures/ -> EDA, eğitim eğrileri, confusion matrix, attention ağırlıkları (PNG)
"""

import os, re, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ======================================================================
# AYARLAR
# ======================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42
DATA_DIR = os.path.join(BASE_DIR, "data")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
MAX_SEQ_LEN = 100
EMBED_DIM = 128

KATEGORILER = ["World", "Sports", "Business", "Sci/Tech"]
NUM_CLASSES = 4

os.makedirs(FIGURES_DIR, exist_ok=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def baslik(metin):
    print("\n" + "=" * 60)
    print(f"  {metin}")
    print("=" * 60)


# ======================================================================
# 1 — VERİ YÜKLEME
# ======================================================================

def load_data():
    """AG News verisini CSV'den yükler, yoksa sentetik fallback oluşturur."""
    path = os.path.join(DATA_DIR, "ag_news.csv")
    if not os.path.exists(path):
        print(f"  [!] Veri bulunamadı: {path}")
        print("  HuggingFace: fancyzhx/ag_news")
        print("  Sentetik veri ile devam ediliyor...")
        np.random.seed(SEED)
        texts = {
            0: ["world leaders meet for climate summit today"],
            1: ["team wins championship in final match"],
            2: ["stock market rises today showing gains"],
            3: ["ai breakthrough announced by researchers"],
        }
        rows = []
        for lbl in range(4):
            for t in texts[lbl]:
                for _ in range(15000):
                    rows.append({"text": t, "label": lbl})
        return pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return pd.read_csv(path)


def tokenize(text):
    text = re.sub(r'[^a-z0-9\s]', ' ', str(text).lower())
    return re.sub(r'\s+', ' ', text).strip().split()


# ======================================================================
# 2 — VERİ HAZIRLAMA
# ======================================================================

def veri_hazirla(df):
    """Tokenize, sözlük oluştur, kodla, train/test böl."""
    df["tokens"] = df["text"].apply(tokenize)

    freq = Counter(t for tlist in df["tokens"] for t in tlist)
    word_list = [(w, c) for w, c in freq.items() if c >= 5]
    vocab = {"<PAD>": 0, "<UNK>": 1}
    vocab.update({w: i + 2 for i, (w, _) in enumerate(word_list)})
    vocab_size = len(vocab)
    print(f"  Sözlük boyutu: {vocab_size:,}")

    def encode(tokens):
        ids = [vocab.get(t, 1) for t in tokens[:MAX_SEQ_LEN]]
        return ids + [0] * (MAX_SEQ_LEN - len(ids))

    df["encoded"] = df["tokens"].apply(encode)
    df["seq_len"] = df["tokens"].apply(len)
    print(f"  Ortalama token: {df['seq_len'].mean():.1f}")

    X = np.array(df["encoded"].tolist())
    y = df["label"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"  Eğitim: {len(X_tr):,} | Test: {len(X_te):,}")

    return df, vocab, vocab_size, X_tr, X_te, y_tr, y_te


# ======================================================================
# 3 — EDA
# ======================================================================

def eda_grafikler(df):
    """Sınıf dağılımı, token uzunluğu ve en sık kelimeler."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("AG News — Keşifsel Analiz", fontsize=14, fontweight="bold")

    counts = df["label"].value_counts().sort_index()
    axes[0].bar(KATEGORILER, counts.values, color=sns.color_palette("Set2", 4))
    axes[0].set_title("Sınıf Dağılımı"); axes[0].set_ylabel("Haber Sayısı")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 200, str(v), ha="center", fontsize=9)

    axes[1].hist(df["seq_len"], bins=50, color="#4C72B0", edgecolor="white")
    axes[1].axvline(MAX_SEQ_LEN, color="red", ls="--", label=f"MAX={MAX_SEQ_LEN}")
    axes[1].set_title("Token Dağılımı"); axes[1].set_xlabel("Token"); axes[1].legend()

    sw = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "is",
          "it", "this", "that", "with", "from", "by", "be", "are", "was", "were",
          "not", "but", "or", "as", "its", "their", "will", "would", "could"}
    all_tok = [t for tl in df["tokens"] for t in tl if t not in sw and len(t) > 2]
    top15 = Counter(all_tok).most_common(15)
    axes[2].barh([w for w, _ in top15[::-1]], [c for _, c in top15[::-1]], color="#4C72B0")
    axes[2].set_title("En Sık 15 Kelime"); axes[2].set_xlabel("Frekans")

    plt.tight_layout()
    yol = os.path.join(FIGURES_DIR, "01_eda.png")
    plt.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [kaydedildi] {yol}")


# ======================================================================
# 4 — DATASET SINIFI
# ======================================================================

class NewsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X), dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ======================================================================
# 5 — ATTENTION MODELİ
# ======================================================================

class BahdanauAttention(nn.Module):
    """Additive (Bahdanau-style) Attention.

    Her token pozisyonu için bir ağırlık hesaplar.
    Yüksek ağırlık → modelin o token'a odaklandığı anlamına gelir.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, lstm_outputs):
        enerjiler = self.v(torch.tanh(self.attn(lstm_outputs)))
        agirliklar = F.softmax(enerjiler.squeeze(-1), dim=1)
        context = torch.bmm(agirliklar.unsqueeze(1), lstm_outputs)
        return context.squeeze(1), agirliklar


class BiLSTMAttention(nn.Module):
    """Embedding → BiLSTM → Bahdanau Attention → Linear sınıflandırıcı.

    BiLSTM: İleri + geri yönlü LSTM → her token için çift yönlü bağlam.
    Attention: Hangi token'ların önemli olduğunu öğrenir.
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        self.attention = BahdanauAttention(hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)             # (batch, seq, hidden*2)
        context, attn_weights = self.attention(lstm_out)
        out = self.dropout(context)
        return self.fc(out), attn_weights


# ======================================================================
# 6 — EĞİTİM
# ======================================================================

def egit(X_tr, X_te, y_tr, y_te, vocab_size):
    """BiLSTM + Attention modelini eğitir."""
    baslik("BiLSTM + ATTENTION EĞİTİMİ")

    tr_loader = DataLoader(NewsDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    te_loader = DataLoader(NewsDataset(X_te, y_te), batch_size=BATCH_SIZE)

    model = BiLSTMAttention(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
    model = model.to(DEVICE)
    print(f"  Cihaz: {DEVICE}")
    print(f"  Model: Embedding({vocab_size}→{EMBED_DIM}) + "
          f"BiLSTM({EMBED_DIM}→{HIDDEN_SIZE}) + Attention + Linear → {NUM_CLASSES}")
    print(f"  Parametre: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses, val_accs, val_f1s = [], [], []

    for epoch in range(1, EPOCHS + 1):
        model.train(); epoch_loss = 0
        for Xb, yb in tqdm(tr_loader, desc=f"  Epoch {epoch}/{EPOCHS}", leave=False):
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(yb)
        scheduler.step()

        avg_loss = epoch_loss / len(tr_loader.dataset)
        train_losses.append(avg_loss)

        model.eval(); preds, labs = [], []
        with torch.no_grad():
            for Xb, yb in te_loader:
                logits, _ = model(Xb.to(DEVICE))
                preds.extend(torch.argmax(logits, 1).cpu().tolist())
                labs.extend(yb.tolist())
        acc = accuracy_score(labs, preds) * 100
        f1_m = f1_score(labs, preds, average="macro")
        val_accs.append(acc); val_f1s.append(f1_m)
        print(f"    Loss: {avg_loss:.4f}  Acc: {acc:.1f}%  F1: {f1_m:.4f}")

    return model, te_loader, train_losses, val_accs, val_f1s


# ======================================================================
# 7 — GRAFİKLER
# ======================================================================

def egitim_egrileri(train_losses, val_accs, val_f1s):
    """Eğitim kaybı ve doğruluk/F1 eğrileri."""
    BG, BLUE, GREEN, ORANGE = "#1e1e2e", "#5c7cfa", "#40c057", "#ff922b"

    with plt.style.context("dark_background"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor(BG)
        for ax in (ax1, ax2): ax.set_facecolor(BG)

        ax1.plot(range(EPOCHS), train_losses, color=BLUE, lw=2)
        ax1.set_title("Eğitim Kaybı", color="white")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Kayıp")
        ax1.tick_params(colors="white"); ax1.grid(color="#3a3a5c")

        ax2.plot(range(EPOCHS), val_accs, color=GREEN, lw=2, label="Accuracy")
        ax2.plot(range(EPOCHS), val_f1s, color=ORANGE, lw=2, ls="--", label="F1")
        ax2.axhline(25, color="gray", ls=":", label="Random")
        ax2.set_title("Doğruluk ve F1", color="white")
        ax2.set_xlabel("Epoch"); ax2.tick_params(colors="white")
        ax2.grid(color="#3a3a5c")
        ax2.legend(facecolor=BG, edgecolor="#3a3a5c", labelcolor="white")
        ax2.set_ylim(0, 100)

        plt.tight_layout()
        yol = os.path.join(FIGURES_DIR, "02_training_curves.png")
        plt.savefig(yol, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close()
    print(f"  [kaydedildi] {yol}")


def degerlendir(model, test_loader, X_te, y_te):
    """Test seti üzerinde nihai rapor, confusion matrix ve attention görselleştirme."""
    baslik("TEST SONUÇLARI")

    model.eval(); all_preds, all_labels = [], []
    all_attn_weights = []
    with torch.no_grad():
        for Xb, yb in test_loader:
            logits, attn_w = model(Xb.to(DEVICE))
            all_preds.extend(torch.argmax(logits, 1).cpu().tolist())
            all_labels.extend(yb.tolist())
            if not all_attn_weights:
                all_attn_weights.append(attn_w[0].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1_m = f1_score(all_labels, all_preds, average="macro")
    print(f"  Doğruluk: {acc:.4f} | F1 (macro): {f1_m:.4f}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=KATEGORILER, zero_division=0)}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=KATEGORILER, yticklabels=KATEGORILER)
    plt.title("Confusion Matrix — BiLSTM + Attention")
    plt.ylabel("Gerçek"); plt.xlabel("Tahmin")
    plt.tight_layout()
    yol = os.path.join(FIGURES_DIR, "03_confusion_matrix.png")
    plt.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [kaydedildi] {yol}")

    # Attention ağırlıkları
    if all_attn_weights:
        attn_w = all_attn_weights[0][:MAX_SEQ_LEN]
        plt.figure(figsize=(10, 3))
        plt.bar(range(len(attn_w)), attn_w, color="#ff6b6b", alpha=0.7)
        plt.title("Attention Ağırlıkları — Model Hangi Token'lara Odaklanıyor?")
        plt.xlabel("Token Pozisyonu"); plt.ylabel("Ağırlık")
        plt.tight_layout()
        yol = os.path.join(FIGURES_DIR, "04_attention_weights.png")
        plt.savefig(yol, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [kaydedildi] {yol}")

    # Örnek tahminler
    print("\n  Örnek Tahminler:")
    model.eval()
    for idx in np.random.choice(len(X_te), 5, replace=False):
        seq = torch.tensor([X_te[idx]], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            logits, _ = model(seq)
            p = torch.softmax(logits, 1)
            pred = torch.argmax(p, 1).item()
        flag = "✓" if pred == y_te[idx] else "✗"
        print(f"    {flag} Gerçek: {KATEGORILER[y_te[idx]]:<10} → "
              f"Tahmin: {KATEGORILER[pred]:<10} (%{p[0, pred].item()*100:.0f})")


# ======================================================================
# ANA PROGRAM
# ======================================================================

def main():
    print("=" * 60)
    print("    BiLSTM + ATTENTION — AG NEWS SINIFLANDIRMASI")
    print("    Embedding → BiLSTM → Bahdanau Attention → Linear")
    print("=" * 60)

    baslik("Veri Yükleniyor")
    df = load_data()
    print(f"  Veri: {len(df):,} haber, {df['label'].nunique()} kategori")

    baslik("Veri Hazırlanıyor")
    df, vocab, vocab_size, X_tr, X_te, y_tr, y_te = veri_hazirla(df)

    eda_grafikler(df)

    model, test_loader, train_losses, val_accs, val_f1s = egit(
        X_tr, X_te, y_tr, y_te, vocab_size
    )

    egitim_egrileri(train_losses, val_accs, val_f1s)
    degerlendir(model, test_loader, X_te, y_te)

    print("\n" + "=" * 60)
    print(f"    TAMAMLANDI — Görseller '{FIGURES_DIR}/' klasöründe.")
    print("=" * 60)


if __name__ == "__main__":
    main()
