"""
AG News Sınıflandırması — LSTM (PyTorch)
=========================================
Veri seti : AG News (4 kategorili haber)
Görev     : 4 sınıflı haber sınıflandırması
Model     : LSTM — Vanilla RNN'deki vanishing gradient sorununu çözer
"""

import os, re, random, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42; DATA_DIR = os.path.join(BASE_DIR, "data"); FIGURES_DIR = os.path.join(BASE_DIR, "figures")
BATCH_SIZE = 64; EPOCHS = 10; LR = 1e-3; HIDDEN_SIZE = 128; NUM_LAYERS = 2
MAX_SEQ_LEN = 100; EMBED_DIM = 128
LABELS = ["World", "Sports", "Business", "Sci/Tech"]; NUM_CLASSES = 4
os.makedirs(FIGURES_DIR, exist_ok=True)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {DEVICE}")

def load_data():
    path = os.path.join(DATA_DIR, "ag_news.csv")
    if not os.path.exists(path):
        print(f"\n[!] Veri bulunamadı: {path}")
        print("    1. https://huggingface.co/datasets/fancyzhx/ag_news")
        print("    from datasets import load_dataset")
        print("    load_dataset('fancyzhx/ag_news', split='train').to_pandas().to_csv('data/ag_news.csv')")
        print("\n[!] Fallback: sentetik veri...")
        np.random.seed(SEED)
        texts = {0: ["world leaders meet for climate summit", "earthquake hits japan"], 1: ["team wins championship", "player scores hat trick"], 2: ["stock market rises today", "company reports profits"], 3: ["ai breakthrough announced", "new smartphone released"]}
        rows = []
        for lbl in range(4):
            for t in texts[lbl]:
                for _ in range(15000):
                    rows.append({"text": t, "label": lbl})
        return pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return pd.read_csv(path)

df = load_data()
print(f"Veri: {len(df)} haber, {df['label'].nunique()} kategori")
print(f"Dağılım: {df['label'].value_counts().sort_index().to_dict()}")

def tokenize(text):
    text = re.sub(r'[^a-z0-9\s]', ' ', str(text).lower())
    return re.sub(r'\s+', ' ', text).strip().split()

df["tokens"] = df["text"].apply(tokenize)
freq = Counter(t for tlist in df["tokens"] for t in tlist)
word_list = [(w, c) for w, c in freq.items() if c >= 5]
vocab = {"<PAD>": 0, "<UNK>": 1}
vocab.update({w: i+2 for i, (w, _) in enumerate(word_list)})
VOCAB_SIZE = len(vocab)
print(f"Sözlük: {VOCAB_SIZE}")

def encode(tokens):
    ids = [vocab.get(t, 1) for t in tokens[:MAX_SEQ_LEN]]
    return ids + [0] * (MAX_SEQ_LEN - len(ids))

df["encoded"] = df["tokens"].apply(encode)
df["seq_len"] = df["tokens"].apply(len)
print(f"Ortalama token: {df['seq_len'].mean():.1f}")

# EDA
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("AG News — Keşifsel Analiz", fontsize=14, fontweight="bold")
counts = df["label"].value_counts().sort_index()
axes[0].bar(LABELS, counts.values, color=sns.color_palette("Set2", 4))
axes[0].set_title("Sınıf Dağılımı"); axes[0].set_ylabel("Haber Sayısı")
for i, v in enumerate(counts.values):
    axes[0].text(i, v+200, str(v), ha="center", fontsize=9)
axes[1].hist(df["seq_len"], bins=50, color="#4C72B0", edgecolor="white")
axes[1].axvline(MAX_SEQ_LEN, color="red", ls="--", label=f"MAX={MAX_SEQ_LEN}")
axes[1].set_title("Token Dağılımı"); axes[1].set_xlabel("Token"); axes[1].legend()
sw = {"the","a","an","in","on","at","to","for","of","and","is","it","this","that","with","from","by","be","are","was","were","has","have","had","not","but","or","as","its","their","will","would","could","should","all","more","very","just","also","about","than","them","they","after","before","said","new"}
all_tok = [t for tl in df["tokens"] for t in tl if t not in sw and len(t)>2]
top15 = Counter(all_tok).most_common(15)
axes[2].barh([w for w,_ in top15[::-1]], [c for _,c in top15[::-1]], color="#4C72B0")
axes[2].set_title("En Sık 15 Kelime"); axes[2].set_xlabel("Frekans")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "01_eda.png"), dpi=150, bbox_inches="tight")
plt.close(); print("-> figures/01_eda.png")

# Dataset
class NewsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X), dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

X = np.array(df["encoded"].tolist()); y = df["label"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
tr_loader = DataLoader(NewsDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
te_loader = DataLoader(NewsDataset(X_te, y_te), batch_size=BATCH_SIZE)
print(f"Eğitim: {len(X_tr)} | Test: {len(X_te)}")

# LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True, dropout=0.3 if num_layers>1 else 0)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        emb = self.embedding(x)
        _, (hn, _) = self.lstm(emb)
        out = self.dropout(hn[-1])
        return self.fc(out)

model = LSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(DEVICE)
print(f"\nModel parametre: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training
print("\nEğitim...")
train_losses, val_accs, val_f1s = [], [], []
for epoch in range(1, EPOCHS+1):
    model.train(); epoch_loss = 0
    for Xb, yb in tqdm(tr_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); epoch_loss += loss.item() * len(yb)
    scheduler.step()
    avg_loss = epoch_loss / len(tr_loader.dataset); train_losses.append(avg_loss)
    model.eval(); preds, labs = [], []
    with torch.no_grad():
        for Xb, yb in te_loader:
            preds.extend(torch.argmax(model(Xb.to(DEVICE)), 1).cpu().tolist())
            labs.extend(yb.tolist())
    acc = accuracy_score(labs, preds)*100; f1_m = f1_score(labs, preds, average="macro")
    val_accs.append(acc); val_f1s.append(f1_m)
    print(f"  Epoch {epoch:>2}  Loss: {avg_loss:.4f}  Acc: {acc:.2f}%  F1: {f1_m:.4f}")

# Figures
BG="#1e1e2e"; GRID="#3a3a5c"
with plt.style.context("dark_background"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2): ax.set_facecolor(BG)
    ax1.plot(range(EPOCHS), train_losses, color="#5c7cfa", lw=2)
    ax1.set_title("Eğitim Kaybı", color="white"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Kayıp"); ax1.tick_params(colors="white"); ax1.grid(color=GRID)
    ax2.plot(range(EPOCHS), val_accs, color="#40c057", lw=2, label="Accuracy")
    ax2.plot(range(EPOCHS), val_f1s, color="#ff922b", lw=2, ls="--", label="F1 (macro)")
    ax2.axhline(25, color="gray", ls=":", label="Random")
    ax2.set_title("Doğruluk ve F1", color="white"); ax2.set_xlabel("Epoch"); ax2.tick_params(colors="white"); ax2.grid(color=GRID); ax2.legend(facecolor=BG, edgecolor=GRID, labelcolor="white"); ax2.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "02_training_curves.png"), dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
print("-> figures/02_training_curves.png")

# Evaluation
model.eval(); all_preds, all_labels = [], []
with torch.no_grad():
    for Xb, yb in te_loader:
        all_preds.extend(torch.argmax(model(Xb.to(DEVICE)), 1).cpu().tolist())
        all_labels.extend(yb.tolist())
acc = accuracy_score(all_labels, all_preds)
f1_m = f1_score(all_labels, all_preds, average="macro")
print(f"\n─── Test ───────────────────────────────")
print(f"Accuracy: {acc:.4f} | F1 (macro): {f1_m:.4f}")
print(f"\n{classification_report(all_labels, all_preds, target_names=LABELS, zero_division=0)}")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
plt.title("Confusion Matrix — LSTM"); plt.ylabel("Gerçek"); plt.xlabel("Tahmin")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "03_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close(); print("-> figures/03_confusion_matrix.png")

print("\nÖrnek Tahminler:")
model.eval()
for idx in np.random.choice(len(X_te), 5, replace=False):
    seq = torch.tensor([X_te[idx]], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        p = torch.softmax(model(seq), 1)
        pred = torch.argmax(p, 1).item()
    flag = "✓" if pred == y_te[idx] else "✗"
    print(f"  {flag} Gerçek: {LABELS[y_te[idx]]:<10} Tahmin: {LABELS[pred]:<10} (p={p[0,pred].item():.3f})")

print("\nTamamlandı.")
