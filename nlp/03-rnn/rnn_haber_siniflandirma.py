"""
Türkçe Haber Sınıflandırması — Vanilla RNN (PyTorch)
======================================================
Veri seti : TTC-4900 (Türkçe Haber Kategorizasyonu)
             Kaggle: savasy/ttc4900
Görev     : 7 kategorili haber sınıflandırması
Model     : Tek katmanlı Vanilla RNN (nn.RNN)
             Embedding → nn.RNN(tanh) → Linear(7) → CrossEntropyLoss
"""

import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

# ─────────────────────────────────────────────
# 0. Ayarlar
# ─────────────────────────────────────────────
SEED           = 42
DATA_DIR       = "data"
FIGURES_DIR    = "figures"
BATCH_SIZE     = 32
EPOCHS         = 20
LR             = 2e-3
HIDDEN_SIZE    = 128
NUM_LAYERS     = 1
MAX_SEQ_LEN    = 50     # ilk 50 token (haber kategorisi genelde ilk cümlede bellidir)
TEST_SIZE      = 0.2

CATEGORIES = {
    0: "siyaset", 1: "ekonomi", 2: "kultur",
    3: "saglik",  4: "spor",    5: "teknoloji",
    6: "dunya"
}
NUM_CLASSES = len(CATEGORIES)

os.makedirs(FIGURES_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {DEVICE}")


# ─────────────────────────────────────────────
# 1. Veri Yükleme
# ─────────────────────────────────────────────
def load_data():
    csv_path = os.path.join(DATA_DIR, "ttc4900.csv")

    if not os.path.exists(csv_path):
        print(f"\n[!] Veri dosyası bulunamadı: {csv_path}")
        print(f"[!] Lütfen TTC-4900 veri setini indirin:")
        print(f"    1. https://www.kaggle.com/datasets/savasy/ttc4900")
        print(f"    2. 'ttc4900.csv' dosyasını '{DATA_DIR}/' klasörüne koyun")
        print(f"    3. Veya HuggingFace'den yükleyin:")
        print(f"       from datasets import load_dataset")
        print(f"       df = load_dataset('savasy/ttc4900', split='train').to_pandas()")
        print(f"       df.to_csv('{DATA_DIR}/ttc4900.csv', index=False)")
        print(f"\n[!] Fallback: sentetik Türkçe haber verisi oluşturuluyor...\n")
        return create_fallback_data()

    df = pd.read_csv(csv_path)
    print(f"Veri yüklendi: {len(df)} haber, {df['category'].nunique()} kategori")
    print(f"Kategoriler: {df['category'].value_counts().sort_index().to_dict()}")
    return df


def create_fallback_data():
    np.random.seed(SEED)
    fallback_texts = {
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
    for cat_id, texts in fallback_texts.items():
        for _ in range(140):
            for t in texts:
                rows.append({"category": cat_id, "category_name": CATEGORIES[cat_id], "text": t})
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df = df.groupby("category").head(700).reset_index(drop=True)
    print(f"  Fallback veri: {len(df)} haber")
    print(f"  Kategoriler: {df['category'].value_counts().sort_index().to_dict()}")
    return df


# ─────────────────────────────────────────────
# 2. Tokenizasyon ve Kodlama
# ─────────────────────────────────────────────
def turkish_tokenize(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zöçşığüâîû0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    return tokens


def build_vocab(all_tokens, min_freq=5):
    freq = {}
    for tokens in all_tokens:
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
    vocab = {}
    idx = 2
    for w, c in freq.items():
        if c >= min_freq:
            vocab[w] = idx
            idx += 1
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab


def encode(tokens, vocab, max_len):
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids


# ─────────────────────────────────────────────
# 3. Veri Yükleme ve İşleme
# ─────────────────────────────────────────────
print("\nVeri yükleniyor...")
df = load_data()

# Etiketler zaten 0-6
print(f"\nTokenizasyon yapılıyor...")
df["tokens"] = df["text"].apply(turkish_tokenize)

print("Sözlük oluşturuluyor...")
vocab = build_vocab(df["tokens"].tolist())
VOCAB_SIZE = len(vocab)
print(f"  Sözlük boyutu: {VOCAB_SIZE}")

print("Diziler kodlanıyor...")
df["encoded"] = df["tokens"].apply(lambda x: encode(x, vocab, MAX_SEQ_LEN))
df["seq_len"] = df["tokens"].apply(len)

print(f"  Ortalama token sayısı: {df['seq_len'].mean():.1f}")
print(f"  Maks token sayısı: {df['seq_len'].max()}")


# ─────────────────────────────────────────────
# 4. EDA Görselleri
# ─────────────────────────────────────────────
print("\nEDA görselleri üretiliyor...")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("TTC-4900 Türkçe Haber Veri Seti — Keşifsel Analiz", fontsize=14, fontweight="bold")

cat_names = [CATEGORIES[i] for i in range(NUM_CLASSES)]
counts = df["category"].value_counts().sort_index()
colors_bar = sns.color_palette("Set2", NUM_CLASSES)
axes[0].bar(cat_names, counts.values, color=colors_bar)
axes[0].set_title("Sınıf Dağılımı")
axes[0].set_ylabel("Haber Sayısı")
axes[0].tick_params(axis="x", rotation=45)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 5, str(v), ha="center", fontsize=9)

seq_lens = df["seq_len"]
axes[1].hist(seq_lens, bins=40, color="#4C72B0", edgecolor="white")
axes[1].axvline(MAX_SEQ_LEN, color="red", linestyle="--", label=f"MAX_SEQ_LEN={MAX_SEQ_LEN}")
axes[1].set_title("Token Sayısı Dağılımı")
axes[1].set_xlabel("Token Sayısı")
axes[1].set_ylabel("Haber Sayısı")
axes[1].legend()

# En sık görülen 15 kelime (Türkçe stop words hariç)
from collections import Counter
stop_words = {"bir", "bu", "da", "de", "daha", "en", "ile", "için", "ise",
              "kadar", "olan", "olarak", "olduğu", "oldu", "üzere", "ve",
              "veya", "ya", "ne", "mi", "mu", "mı", "değil", "çok", "her",
              "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
              "m", "n", "o", "p", "r", "s", "t", "u", "v", "y", "z",
              "ile", "ve", "ya da", "gibi", "ki", "de", "da", "dahi",
              "ama", "ancak", "fakat", "lakin", "çünkü", "eğer", "yoksa",
              "ayrıca", "üstelik", "bile", "hatta"}
all_tokens = []
for t in df["tokens"]:
    all_tokens.extend(t)
freq = Counter(all_tokens)
freq_filtered = {w: c for w, c in freq.items() if w not in stop_words and len(w) > 2}
top_words = Counter(freq_filtered).most_common(15)
labels_w, vals = zip(*top_words)
axes[2].barh(labels_w[::-1], vals[::-1], color="#4C72B0")
axes[2].set_title("En Sık Görülen 15 Kelime (stop words hariç)")
axes[2].set_xlabel("Frekans")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "01_eda.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> figures/01_eda.png kaydedildi")


# ─────────────────────────────────────────────
# 5. Dataset & DataLoader
# ─────────────────────────────────────────────
class HaberDataset(Dataset):
    def __init__(self, encodings, labels):
        self.X = torch.tensor(np.array(encodings), dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


X = np.array(df["encoded"].tolist())
y = df["category"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)

train_loader = DataLoader(HaberDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(HaberDataset(X_test, y_test),
                         batch_size=BATCH_SIZE, shuffle=False)

print(f"\nEğitim: {len(X_train)} | Test: {len(X_test)}")


# ─────────────────────────────────────────────
# 6. Vanilla RNN Modeli
# ─────────────────────────────────────────────
class VanillaRNN(nn.Module):
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
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        out = self.dropout(hidden[-1])
        out = self.fc(out)
        return out


EMBED_DIM = 128
model = VanillaRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(DEVICE)
print(f"\nModel:\n{model}")
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Eğitilebilir parametre: {total_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ─────────────────────────────────────────────
# 7. Eğitim
# ─────────────────────────────────────────────
print("\nEğitim başlıyor...")
train_losses, val_accs, val_f1s = [], [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
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

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y_batch.tolist())

    acc = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    val_accs.append(acc)
    val_f1s.append(f1)
    print(f"  Epoch {epoch:>2}/{EPOCHS}  Loss: {avg_loss:.4f}  Acc: {acc:.2f}%  F1: {f1:.4f}")


# ─────────────────────────────────────────────
# 8. Eğitim Eğrileri
# ─────────────────────────────────────────────
BG   = "#1e1e2e"
GRID = "#3a3a5c"
BLUE = "#5c7cfa"
GREEN = "#40c057"
ORANGE = "#ff922b"

with plt.style.context("dark_background"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)

    ax1.plot(range(EPOCHS), train_losses, color=BLUE, linewidth=2)
    ax1.set_title("Eğitim Kaybı (CrossEntropyLoss)", color="white", fontsize=12)
    ax1.set_xlabel("Epoch", color="white")
    ax1.set_ylabel("Kayıp", color="white")
    ax1.tick_params(colors="white")
    ax1.grid(color=GRID, linewidth=0.7)
    for spine in ax1.spines.values():
        spine.set_edgecolor(GRID)

    baseline = (max(np.bincount(y_test)) / len(y_test)) * 100
    ax2.plot(range(EPOCHS), val_accs, color=GREEN, linewidth=2)
    ax2.plot(range(EPOCHS), val_f1s, color=ORANGE, linewidth=2, linestyle="--")
    ax2.axhline(baseline, color="gray", linestyle=":",
                label=f"Çoğunluk tahmini ({baseline:.0f}%)")
    ax2.set_title("Doğruluk ve F1 Skoru (%)", color="white", fontsize=12)
    ax2.set_xlabel("Epoch", color="white")
    ax2.set_ylabel("Skor (%)", color="white")
    ax2.set_ylim(0, 100)
    ax2.tick_params(colors="white")
    ax2.grid(color=GRID, linewidth=0.7)
    ax2.legend(facecolor=BG, edgecolor=GRID, labelcolor="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor(GRID)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "02_training_curves.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
print("  -> figures/02_training_curves.png kaydedildi")


# ─────────────────────────────────────────────
# 9. Final Değerlendirme
# ─────────────────────────────────────────────
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        logits = model(X_batch)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(y_batch.tolist())

acc = accuracy_score(all_labels, all_preds)
f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

print("\n─── Test Sonuçları ───────────────────────────────")
print(f"  Accuracy       : {acc:.4f}")
print(f"  F1 (macro)     : {f1_macro:.4f}")
print(f"  F1 (weighted)  : {f1_weighted:.4f}")
print(f"\n{classification_report(
    all_labels, all_preds,
    target_names=[CATEGORIES[i] for i in range(NUM_CLASSES)],
    zero_division=0
)}")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[CATEGORIES[i] for i in range(NUM_CLASSES)],
            yticklabels=[CATEGORIES[i] for i in range(NUM_CLASSES)],
            ax=ax)
ax.set_title("Confusion Matrix — Vanilla RNN", fontsize=13, fontweight="bold")
ax.set_ylabel("Gerçek")
ax.set_xlabel("Tahmin")
ax.tick_params(axis="x", rotation=45)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "03_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> figures/03_confusion_matrix.png kaydedildi")


# ─────────────────────────────────────────────
# 10. Örnek Tahminler
# ─────────────────────────────────────────────
print("\n─── Örnek Tahminler ─────────────────────────────")
test_df = df.iloc[y_test.argsort()[:5]] if hasattr(y_test, 'argsort') else df.sample(5, random_state=SEED)
sample_indices = np.random.choice(len(X_test), 5, replace=False)

model.eval()
correct_count = 0
for idx in sample_indices:
    seq_tensor = torch.tensor([X_test[idx]], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits = model(seq_tensor)
        prob = torch.softmax(logits, dim=1)
        pred_id = torch.argmax(logits, dim=1).item()
        confidence = prob[0, pred_id].item()

    true_id = y_test[idx]
    pred_label = CATEGORIES[pred_id]
    true_label = CATEGORIES[true_id]
    flag = "✓" if pred_id == true_id else "✗"
    if pred_id == true_id:
        correct_count += 1
    print(f"  {flag}  Gerçek: {true_label:<10} Tahmin: {pred_label:<10} (p={confidence:.3f})")

print(f"\n{correct_count}/5 doğru tahmin")
print("\nTamamlandı. Görseller 'figures/' klasöründe.")
