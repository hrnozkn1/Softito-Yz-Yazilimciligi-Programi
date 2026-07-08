# 📱 Telefon Fiyat Segmenti Sınıflandırma Analizi

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Bu proje, farklı telefon özelliklerine dayanarak telefonların fiyat segmentlerini (Ucuz, Orta, Pahalı, Çok Pahalı) sınıflandırmak için bir makine öğrenimi modeli geliştirmeyi ve değerlendirmeyi amaçlamaktadır.

---

## 📊 Veri Seti

Projede `train.csv` (eğitim ve doğrulama) ve `test.csv` (nihai değerlendirme) olmak üzere iki ana veri seti kullanılmıştır.

### Sütunlar (Özellikler)

| Özellik | Açıklama |
| :--- | :--- |
| `battery_power` | Telefonun depolayabileceği toplam enerji (mAh) |
| `blue` | Bluetooth olup olmadığı (1: evet, 0: hayır) |
| `clock_speed` | Mikroişlemcinin hızı |
| `dual_sim` | Çift SIM desteği olup olmadığı (1: evet, 0: hayır) |
| `fc` | Ön kamera megapikseli |
| `four_g` | 4G desteği olup olmadığı (1: evet, 0: hayır) |
| `int_memory` | Dahili bellek (GB) |
| `m_dep` | Mobil derinlik (cm) |
| `mobile_wt` | Telefonun ağırlığı |
| `n_cores` | İşlemci çekirdek sayısı |
| `pc` | Birincil kamera megapikseli |
| `px_height` | Piksel çözünürlüğü yüksekliği |
| `px_width` | Piksel çözünürlüğü genişliği |
| `ram` | Rastgele erişim belleği (MB) |
| `sc_h` | Ekran yüksekliği (cm) |
| `sc_w` | Ekran genişliği (cm) |
| `talk_time` | Tek bir şarjla mümkün olan en uzun konuşma süresi |
| `three_g` | 3G desteği olup olmadığı (1: evet, 0: hayır) |
| `touch_screen` | Dokunmatik ekran olup olmadığı (1: evet, 0: hayır) |
| `wifi` | Wi-Fi desteği olup olmadığı (1: evet, 0: hayır) |

### Hedef Değişken
*   `price_range`: Fiyat aralığı (`0: ucuz`, `1: orta`, `2: pahalı`, `3: çok pahalı`)

---

## ⚙️ Metodoloji

*   **Veri Hazırlığı:** `train.csv` veri seti eğitim ve doğrulama kümelerine (`X_train`, `X_val`, `y_train`, `y_val`) bölünmüştür. `test.csv` ise nihai test için `X_test_final` olarak ayrılmıştır. `test.csv`'deki 'id' sütunu tahmin öncesinde çıkarılmıştır.
*   **Model Eğitimi:** İki farklı sınıflandırma modeli kullanılmıştır:
    *   **Karar Ağacı (Decision Tree):** `criterion='gini'` ve `max_depth=5` parametreleri ile eğitilmiştir.
    *   **Random Forest:** `n_estimators=100` ve `max_depth=8` parametreleri ile eğitilmiştir.
*   **Çapraz Doğrulama:** Her iki model için 5-katlı çapraz doğrulama uygulanarak modellerin genellenebilirliği değerlendirilmiştir.
*   **Hiperparametre Optimizasyonu:** Random Forest modeli için `GridSearchCV` kullanılarak `n_estimators` ve `max_depth` hiperparametreleri optimize edilmiştir. En iyi parametreler bulunmuş ve model bu parametrelerle yeniden eğitilmiştir.
*   **Model Değerlendirme:** Eğitim ve doğrulama setleri üzerinde `classification_report` ve `confusion_matrix` kullanılarak modellerin performansı detaylı bir şekilde analiz edilmiştir. Ayrıca, model derinliğine göre eğitim ve doğrulama doğruluk oranlarını gösteren bir grafik ile overfitting durumu incelenmiştir.
*   **Nihai Tahminler:** Optimize edilmiş Random Forest modeli, daha önce görülmemiş `X_test_final` verisi üzerinde fiyat segmenti tahminleri yapmak için kullanılmıştır.

---

## 📈 Temel Bulgular

*   **Random Forest Performansı:** Optimize edilmiş Random Forest modeli, doğrulama setinde Karar Ağacı modeline göre daha yüksek bir doğruluk oranı sergilemiştir. `GridSearchCV` ile yapılan optimizasyon, modelin performansını daha da artırmıştır.
*   **Overfitting Analizi:** *Decision Tree vs. Random Forest - Overfitting (Karmaşıklık) Grafiği*, Random Forest'ın farklı derinliklerde daha stabil ve genellenebilir bir performans gösterdiğini ortaya koymuştur.
*   **Sınıflandırma Raporları:** Her iki modelinin `classification_report` çıktıları, her fiyat segmenti için hassasiyet, geri çağırma ve f1-skor metriklerini sunarak sınıf bazında performansı detaylandırmıştır.
*   **Karmaşıklık Matrisleri:** Karmaşıklık matrisleri, modellerin hangi fiyat segmentlerini doğru tahmin ettiğini ve hangi segmentler arasında karışıklık yaşadığını görsel olarak göstermiştir.

---

## 🚀 Not Defterini Çalıştırma

Bu not defterini çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. `train.csv` ve `test.csv` dosyalarını not defterinin bulunduğu dizine yerleştirin.
2. Gerekli Python kütüphanelerini (`pandas`, `scikit-learn`, `matplotlib`, `seaborn`) yükleyin:
```bash
pip install pandas scikit-learn matplotlib seaborn
## 📝 Lisans
Bu proje [MIT](LICENSE) lisansı altında lisanslanmıştır.
