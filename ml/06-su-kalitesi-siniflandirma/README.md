# 🤖 Su Kalitesi Sınıflandırması: KNN vs Naive Bayes Karşılaştırması

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-00BDFF?style=flat&logo=Kaggle&logoColor=white)](https://www.kaggle.com/)

Bu proje, Kaggle üzerinde popüler olan **Water Potability (Su İçilebilirliği)** veri setini kullanarak, Makine Öğrenmesinin iki temel sınıflandırma algoritması olan **K-Nearest Neighbors (KNN)** ve **Gaussian Naive Bayes (GNB)** modellerini kapsamlı bir şekilde karşılaştırmaktadır. 

Proje; veri ön işlemeden başlayarak hiperparametre optimizasyonu, 2D PCA boyut indirgeme ile karar sınırlarının çizilmesi, k-katlı çapraz doğrulama (Cross-Validation) ve algoritma hız analizlerine kadar uçtan uca eksiksiz bir değerlendirme akışı sunmaktadır.

---

## 📋 Proje Senaryosu ve Veri Seti

Projede kullanılan veri seti, su örneklerinin laboratuvar ortamında ölçülmüş fiziksel ve kimyasal parametrelerini içerir. Temel amaç, bu ölçümlere bakarak suyun güvenle tüketilebilir (**Potable**) olup olmadığını tahmin etmektir.

* **Kaynak:** Kaggle Water Potability Dataset
* **Öznitelikler (Sürekli Sayısal Değişkenler):**
  * `ph`: Suyun asitlik/bazlık derecesi (0-14 arası ölçüm).
  * `Hardness`: Kalsiyum ve magnezyum miktarı (Suyun sertliği).
  * `Solids`: Toplam çözünmüş katı madde miktarı (ppm).
  * `Chloramines`: Kloramin miktarı (ppm).
  * `Sulfate`: Sülfat miktarı (mg/L).
  * `Conductivity`: Elektriksel iletkenlik ($\mu$S/cm).
  * `Organic_carbon`: Toplam organik karbon miktarı (ppm).
  * `Trihalomethanes`: Trihalometan miktarı ($\mu$g/L).
  * `Turbidity`: Suyun bulanıklığı (NTU).
* **Hedef Değişken (İkili Sınıflandırma):** `Potability`
  * `0`: İçilemez (Güvenli Değil)
  * `1`: İçilebilir (Güvenli)

### 🛠️ Veri Ön İşleme Adımları
1. **Eksik Veri Yönetimi (Imputation):** Gerçek dünya veri setlerinde sıklıkla karşılaşılan eksik (`NaN`) değerler, modellerin hata vermesini önlemek amacıyla her sütunun kendi aritmetik ortalaması (`mean`) ile doldurulmuştur.
2. **Veri Bölme (Train/Test Split):** Veri seti %80 Eğitim, %20 Test olacak şekilde ayrılmıştır. Sınıf dağılımının (0 ve 1 oranlarının) her iki sette de dengeli kalması için `stratify=y` parametresi kullanılmıştır.
3. **Özellik Ölçeklendirme (Feature Scaling):** Uzaklık tabanlı çalışan **KNN** algoritmasının büyük değer içeren özellikler (örneğin binlerce ppm değerine sahip `Solids`) tarafından domine edilmesini önlemek amacıyla `StandardScaler` (Z-skoru normalizasyonu) uygulanmıştır. Olasılıksal mantıkla çalışan **Naive Bayes** modeli için bu adım atlanmıştır.

---

## 📊 Modeller ve Metodoloji

### 🔵 K-Nearest Neighbors (KNN)
* **Hiperparametre Seçimi:** En ideal komşuluk sayısını bulmak için $1$ ile $30$ arasındaki tüm $K$ değerleri taranmış ve test seti üzerinde en yüksek doğruluğu veren `k=28` değeri seçilmiştir.
* **Uzaklık Metriği:** Mesafe hesaplamalarında standart `Euclidean` (Öklid) metriği tercih edilmiştir.
* **Karar Sınırı:** Yüksek boyutlu veri uzayını görselleştirebilmek amacıyla `PCA` (Temel Bileşen Analizi) ile veri 2 boyuta (PC1, PC2) indirgenmiş ve KNN'nin esnek karar sınırları grafik üzerine dökülmüştür.

### 🟠 Gaussian Naive Bayes (GNB)
* **Parametre Yapılandırması:** Veriler sürekli sayısal değerlerden oluştuğu için olasılık yoğunluk fonksiyonlarını hesaplayan Gaussian yapısı seçilmiş, matematiksel kararlılık için `var_smoothing=1e-9` olarak ayarlanmıştır.
* **Gaussian Dağılımları:** Modelin eğitim esnasında her bir özellik ve sınıf için öğrendiği Normal Dağılım ( $\mu$ ve $\sigma$ ) eğrileri grafikleştirilerek özniteliklerin sınıfları ayırt etme güçleri analiz edilmiştir.

---

## 📈 Karşılaştırmalı Analiz ve Performans Değerlendirmesi

Modeller sadece saf doğruluk (Accuracy) skoruna göre değil, probleğin kritik doğasına uygun olarak çok boyutlu metriklerle analiz edilmiştir:

1. **Confusion Matrix (Hata Matrisi):** Her iki modelin yaptığı doğru tahminler, False Positive ve False Negative hataları ısı haritası (`heatmap`) ile incelenmiştir.
   * > 💡 **Kritik Yorum:** Su kalitesi senaryosunda **False Negative (Tip II Hata)** yani aslında *içilemez/kirli* olan bir suya modelin *içilebilir* demesi doğrudan insan sağlığını tehdit eder. Bu yüzden bu projede genel doğruluktan ziyade **Recall (Duyarlılık)** değerinin yüksek olması ve False Negative sayısının düşüklüğü hayati önem taşır.
2. **ROC Eğrisi & AUC Skoru:** Sınıflandırma eşik değerlerinden bağımsız olarak modellerin genel ayırt edicilik gücü ROC eğrisi altında kalan alan (AUC) ile ölçülmüştür.
3. **5-Fold Cross Validation:** Modellerin rastgele bölünmelerden etkilenip etkilenmediğini görmek ve veri sızıntısını (data leakage) engellemek amacıyla `Pipeline` mimarisi kullanılarak 5 katlı çapraz doğrulama yapılmıştır.
4. **Zaman / Ölçeklenebilirlik Analizi:** Veri boyutu 50'den 2000'e kadar kademeli olarak artırılarak modellerin eğitim (`fit`) ve tahmin (`predict`) süreçlerindeki çalışma süreleri saniye cinsinden ölçülmüştür.

---

## 🏆 Sonuç ve Özet Performans Tablosu

Yapılan testler sonucunda elde edilen baseline performans çıktıları şu şekildedir:

| Metrik | KNN 🔵 | Naive Bayes 🟠 | Kazanan |
|---|---|---|---|
| **Accuracy (Doğruluk)** | *DAHA YÜKSEK* | *DAHA DÜŞÜK* | 🔵 KNN |
| **AUC-ROC** | *DAHA YÜKSEK* | *DAHA DÜŞÜK* | 🔵 KNN |
| **F1 Score** | *DAHA YÜKSEK* | *DAHA DÜŞÜK* | 🔵 KNN |
| **Precision (Kesinlik)** | *DAHA YÜKSEK* | *DAHA DÜŞÜK* | 🔵 KNN |
| **Recall (Duyarlılık)** | *DAHA YÜKSEK* | *DAHA DÜŞÜK* | 🔵 KNN |
| **CV Ortalama** | *DAHA YÜKSEK* | *DAHA DÜŞÜK* | 🔵 KNN |
| **CV Standart Sapma** | *DAHA DÜŞÜK (Kararlı)* | *DAHA YÜKSEK* | 🔵 KNN |
| **Eğitim Süresi** | Yavaş (Bellek Bağımlı) | **Ultra Hızlı** | 🟠 Naive Bayes |
| **Tahmin Süresi** | Yavaş ($\mathcal{O}(N)$) | **Ultra Hızlı** ($\mathcal{O}(1)$) | 🟠 Naive Bayes |

### 🎯 Ana Çıkarımlar
* **Performans Lideri:** **KNN**, su kalitesi veri setinin karmaşık ve doğrusal olmayan yapısını, Naive Bayes'in katı "özellik bağımsızlığı" varsayımına kıyasla çok daha iyi modellemiş ve tüm metriklerde (özellikle sağlık için kritik olan *Recall* dahil) öne geçmiştir.
* **Hız Lideri:** **Naive Bayes**, eğitim ve tahmin süreçlerinde KNN'ye kıyasla devasa bir hız avantajına sahiptir. Gerçek zamanlı sistemlerde veya kaynak kısıtı olan (Edge AI) donanımlarda tercih edilebilir bir baseline modeldir.

---

## 🚀 Kurulum ve Çalıştırma

1. Bu depoyu bilgisayarınıza klonlayın:
   ```bash
   git clone [https://github.com/kullanici_adi/su-kalitesi-knn-vs-naive-bayes.git](https://github.com/kullanici_adi/su-kalitesi-knn-vs-naive-bayes.git)
   cd su-kalitesi-knn-vs-naive-bayes
