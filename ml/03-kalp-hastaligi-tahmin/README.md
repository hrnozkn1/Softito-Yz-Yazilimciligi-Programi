# 📊 Kalp Hastalığı Tahmin Modelleri Karşılaştırması

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-📊-🚀)](https://matplotlib.org/)

Bu proje, sağlık sektörünün en kritik ve hayati konularından biri olan **Kalp Hastalığı Teşhisi** (`heart.csv`) üzerinde geleneksel makine öğrenmesi algoritmalarını yarıştırmak, en güvenilir modeli belirlemek ve bu modellerin hangi klinik durumlarda hata yaptığını çözmek amacıyla tasarlanmıştır.

---

## 🎯 Projenin Ana Amacı
Kalp hastalığı gibi insan hayatını doğrudan etkileyen bir senaryoda sadece yüksek doğruluk (**Accuracy**) elde etmek yeterli değildir. Asıl amaç, hasta olan birini sağlıklı sanma hatasını (**False Negative**) sıfıra indirecek en duyarlı (**Recall**) modeli keşfetmek ve bu modelin "kör noktalarını" matematiksel olarak analiz etmektir.

---

## 🗂️ Proje Uygulama Adımları

### 1. Veri Yükleme ve Ön İşleme (Preprocessing)
* **Veri Seti:** `heart.csv` veri seti sisteme yüklenmiş, veri kalitesini bozacak eksik değerler (NaN) kontrol edilmiştir.
* **Ölçeklendirme:** Modellerin varyans farklarından etkilenmemesi ve gradyanların kararlı yakınsaması için tüm özellikler `StandardScaler` ile normalize edilmiştir.
* **Veri Bölme:** Verinin dengeli dağılması adına `stratify` parametresi kullanılarak veri seti **%80 Eğitim** ve **%20 Test** olarak ikiye ayrılmıştır.

### 2. Model Eğitimi ve Karar Sınırları (Teorik Yaklaşım)
Projede veriyi ayırmak için 3 farklı geometrik yaklaşım kullanılmıştır:
* **Lojistik Regresyon:** Verileri doğrusal (Linear) bir çizgiyle ikiye bölmeye çalışır. Baseline modelimizdir.
* **Polinom Lojistik Regresyon (Degree=2):** Mevcut özelliklerin karesini ve birbiriyle etkileşimini alarak veriyi genişletir; böylece bükülebilen, eğrisel karar sınırları çizer.
* **Support Vector Machine (SVM):** RBF (Radial Basis Function) kernel kullanarak veriyi matematiksel olarak yüksek boyutlu bir uzaya taşır ve doğrusal olmayan en karmaşık ilişkileri bile başarıyla çözer.

---

## 📈 Performans Metrikleri ve Karşılaştırma

Eğitilen modeller test seti üzerinde sadece doğruluk oranlarına göre değil, tıbbi hassasiyet kriterlerine göre de değerlendirilmiştir:
* **Accuracy (Doğruluk):** Genel doğru teşhis oranı.
* **Recall (Duyarlılık) 🚨:** Gerçek hastaların kaçını modelin yakalayabildiği (En kritik sağlık metriği).
* **F1-Score:** Precision ve Recall dengesi.
* **AUC-ROC:** Modelin hasta olan ile olmayanı ayırt etme gücü.

### 📊 Görselleştirme Paketi
Modellerin performans farklarını net görmek adına notebook içinde şu grafikler üretilmektedir:
1. Modellerin doğruluk skorlarını kıyaslayan **Bar Grafiği**.
2. Modellerin ayırt edici gücünü gösteren **ROC Eğrileri**.
3. Modellerin net hata sayılarını (Doğru Pozitif, Yanlış Negatif vb.) gösteren **yan yana Confusion Matrix (Karmaşıklık Matrisi)** görselleştirmeleri.

---

## 🔍 Kritik Hata Analizi (Error Analysis)

Projenin en özgün ve ileri düzey mühendislik adımı bu kısımdır. Yapılan testler sonucunda en yüksek performansı gösteren şampiyon model (**SVM**) seçilmiş ve bu modelin **hata yaptığı durumlar** mercek altına alınmıştır:

* **Yanlış Pozitifler (FP - False Positives):** Aslında sağlıklı olduğu halde modelin "Hasta" teşhisi koyduğu uç örnekler ayıklanmıştır.
* **Yanlış Negatifler (FN - False Negatives):** *En hayati risk taşıyan*, gerçekte hasta olduğu halde modelin gözden kaçırıp "Sağlıklı" dediği kritik kişilerin örnekleri çekilmiştir.
* **Klinik İnceleme:** Bu hatalı örneklerin yaş, kolesterol, kan basıncı gibi klinik özelliklerinin ortalamaları ve dağılımları incelenerek, modelin tam olarak hangi hasta profillerinde körleştiği ve hata yapma eğiliminde olduğu bilimsel olarak araştırılmıştır.

---

## 🚀 Kurulum ve Çalıştırma

Projenizi yerelde çalıştırmak için aşağıdaki komutları sırasıyla terminalde koşturabilirsiniz:

```bash
# 1. Depoyu bilgisayarınıza klonlayın
git clone [https://github.com/hrnozkn1/traditional-ml-algorithms-benchmark.git](https://github.com/hrnozkn1/traditional-ml-algorithms-benchmark.git)

# 2. Proje dizinine girin
cd traditional-ml-algorithms-benchmark

# 3. Gerekli kütüphaneleri yükleyin
pip install pandas numpy scikit-learn matplotlib seaborn
