# ⚽ Futbolcu Verileri ile Denetimsiz Öğrenme (Kümeleme Analizi)

Bu proje, `players.csv` veri setinde yer alan futbolcuları yaş, mevki ve milliyet gibi özelliklerine göre gruplandırmak ve benzer oyuncu profillerini keşfetmek amacıyla yapılmış başlangıç seviyesinde bir **Denetimsiz Öğrenme (Unsupervised Learning)** çalışmasıdır.

## 📁 Proje Yapısı

* `data/players.csv`: Analizde kullanılan ham futbolcu veri seti.
* `futbolcu_kumeleme_analizi.ipynb`: Google Colab üzerinde geliştirilen; veri ön işleme, modelleme ve görselleştirme adımlarını içeren kod dosyası.

## 🤖 Kullanılan Yöntemler ve Algoritmalar

Projede verilerin yapısını anlamak ve oyuncuları anlamlı gruplara ayırmak için şu yöntemler uygulanmıştır:

1. **Veri Ön İşleme:** Kategorik verilerin sayısallaştırılması (OneHotEncoder) ve yaş değişkeninin ölçeklendirilmesi (StandardScaler).
2. **K-Means Kümeleme:** En doğru küme sayısını bulmak için *Dirsek (Elbow) Metodu*.
3. **Hiyerarşik Kümeleme (Agglomerative)**
4. **Gaussian Mixture Models (GMM)**
5. **PCA (Temel Bileşen Analizi):** Çok boyutlu oyuncu verilerini grafik üzerinde rahatça görebilmek için 2 boyuta indirgeme ve görselleştirme.

## 🛠️ Kurulum ve Çalıştırma

Gerekli kütüphaneleri yüklemek için terminalinizde şu komutu çalıştırabilirsiniz:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
