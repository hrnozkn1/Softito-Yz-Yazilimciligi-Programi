
# 📈 Hava Durumu API'li Zaman Serisi RNN Modeli

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![API](https://img.shields.io/badge/Data%20Source-Open--Meteo%20API-blueviolet)](https://open-meteo.com/)

Bu proje, ardışık ve zamana bağlı verilerin (Time Series) analizinde kullanılan **Yinelemeli Sinir Ağları (Recurrent Neural Networks - RNN)** mimarisini uçtan uca uygulamak amacıyla geliştirilmiştir. 

Proje, harici bir veri seti dosyasına (`.csv`, `.xlsx` vb.) ihtiyaç duymadan, **Open-Meteo API** üzerinden İstanbul'un son 3 yıllık gerçek günlük maksimum sıcaklık verilerini canlı olarak çekerek çalışır. Model, geçmiş günlerin sıcaklık örüntülerini analiz ederek gelecek günün sıcaklığını tahmin eder.

---

## 🎯 Projenin Ana Amacı
Geleneksel yapay sinir ağları (ANN), girdiler arasındaki zamansal ve sıralı ilişkileri (hafıza faktörünü) koruyamaz. Bu projede amacımız:
1. Zaman serisi verilerini yapay sinir ağlarının işleyebileceği **Lookback (Geçmiş Penceresi)** formuna getirmek.
2. `Simple RNN` katmanı kullanarak verideki mevsimsel döngüleri ve zamansal bağımlılıkları (temporal dependencies) yakalamak.
3. Gerçek çevre verileri üzerinde regresyon tabanlı bir gelecek tahmini (forecasting) mekanizması inşa etmektir.

---

## 🗂️ Proje Uygulama Adımları

### 1. Canlı Veri Hattı (API Integration & Data Pipeline)
* **Canlı Veri Çekimi:** `requests` kütüphanesiyle Open-Meteo arşiv API'sine bağlanılarak İstanbul lokasyonuna ait günlük maksimum sıcaklık verileri otomatik olarak çekilir.
* **Veri Ölçeklendirme:** RNN modelinin kararlı yakınsaması ve eğitim sırasında gradyan patlamalarının/sönümlenmelerinin engellenmesi adına veriler `MinMaxScaler` ile `[0, 1]` aralığına normalize edilir.
* **Zaman Penceresi Tasarımı:** Modelin öğrenmesi için 14 günlük (2 hafta) geçmiş veri penceresi girdi ($X$), 15. günün sıcaklık değeri ise hedef ($y$) olarak etiketlenir.

### 2. RNN Mimarisinin İnşası (PyTorch Sınıf Tasarımı)
* **RNN Katmanı:** `nn.RNN` kullanılarak ardışık verideki her bir zaman adımının gizli durumu (hidden state) bir sonraki adıma aktarılır.
* **Doğrusal Çıktı Katmanı:** Zaman serisinin son adımından elde edilen gizli durum bilgisi, tek bir sürekli değer (sıcaklık tahmini) üretmesi için `nn.Linear` katmanına aktarılır.
* **Kayıp ve Optimizasyon:** Model, regresyon problemlerinin standardı olan **Mean Squared Error (MSE Loss)** fonksiyonu ve **Adam Optimizer** algoritması ile eğitilir.

---

## 📊 Raporlama ve Görselleştirme

Eğitim bittikten sonra test seti (verinin %20'si) üzerindeki tahminler, `MinMaxScaler` ile orijinal Santigrat Derece ($^\circ\text{C}$) formuna geri dönüştürülür. Çıktı olarak şu iki grafik üretilir:
1. **Eğitim Kayıp Grafiği (Loss Trend):** Modelin epoch'lar boyunca hatayı nasıl minimize ettiğinin takibi.
2. **Gerçek vs Tahmin Grafiği:** API'den gelen gerçek İstanbul sıcaklıkları ile RNN modelinin yaptığı tahminlerin zamana bağlı üst üste bindirilmiş kıyası.

---

## 🚀 Kurulum ve Çalıştırma

```bash
# 1. Depoyu bilgisayarınıza klonlayın
git clone [https://github.com/hrnozkn1/time-series-rnn-forecasting.git](https://github.com/hrnozkn1/time-series-rnn-forecasting.git)

# 2. Proje dizinine girin
cd time-series-rnn-forecasting

# 3. Gerekli kütüphaneleri yükleyin
pip install torch pandas numpy matplotlib scikit-learn requests
