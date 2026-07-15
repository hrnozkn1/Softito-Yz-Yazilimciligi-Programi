"""Veri Drift Tespiti: dağılım değişimini yakalama"""
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

os.makedirs("cikti", exist_ok=True)

def ks_test_drift(referans, yeni, alpha=0.05):
    statistic, p_value = stats.ks_2samp(referans, yeni)
    drift_var = p_value < alpha
    return {"istatistik": statistic, "p_degeri": p_value, "drift": drift_var}

def js_divergence(p, q):
    p = np.histogram(p, bins=30, density=True)[0] + 1e-10
    q = np.histogram(q, bins=30, density=True)[0] + 1e-10
    p /= p.sum(); q /= q.sum()
    m = (p + q) / 2
    return (stats.entropy(p, m) + stats.entropy(q, m)) / 2

def wasserstein_distance(a, b):
    return stats.wasserstein_distance(a, b)

def veri_uret(n=1000, ortalama=0, std=1):
    return np.random.normal(ortalama, std, n)

def drift_senaryolari():
    ref = veri_uret(1000, 0, 1)
    print("Drift senaryoları:")

    senaryolar = [
        ("Dağılım aynı", veri_uret(1000, 0, 1)),
        ("Ortalama kaymış", veri_uret(1000, 0.5, 1)),
        ("Varyans artmış", veri_uret(1000, 0, 2)),
        ("Tamamen farklı", veri_uret(1000, 3, 0.5)),
    ]

    sonuclar = []
    for isim, yeni in senaryolar:
        ks = ks_test_drift(ref, yeni)
        js = js_divergence(ref, yeni)
        ws = wasserstein_distance(ref, yeni)
        sonuclar.append({"senaryo": isim, "ks": ks, "js": js, "wasserstein": ws})
        print(f"\n  {isim}:")
        print(f"    KS: p={ks['p_degeri']:.4f} {'DRIFT!' if ks['drift'] else 'yok'}")
        print(f"    JS Divergence: {js:.4f}")
        print(f"    Wasserstein: {ws:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (isim, yeni) in zip(axes.flat, [(n, d) for n, d in [(s[0], s[1]) for s in senaryolar]]):
        ax.hist(ref, bins=30, alpha=0.5, label="Referans", density=True)
        ax.hist(yeni, bins=30, alpha=0.5, label="Yeni", density=True)
        ax.set_title(isim); ax.legend()
    plt.savefig("cikti/drift_dagilimlari.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("\nÇıktı: cikti/drift_dagilimlari.png")

if __name__ == "__main__":
    print("Veri Drift Tespiti\n")
    drift_senaryolari()
