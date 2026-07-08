# Regresyon ile Araba Fiyat Tahmin Analizi

## 📝 Proje Tanımı

Bu proje, çeşitli özelliklere (model yılı, kilometre, yakıt tipi, satıcı türü vb.) dayanarak kullanılmış araba fiyatlarını tahmin etmek amacıyla geliştirilmiştir. Projenin temel amacı, bir regresyon modeli kullanarak araba fiyatları üzerinde etkili olan faktörleri belirlemek ve güvenilir fiyat tahminleri yapabilmektir.

## 📊 Veri Seti

Bu analizde, Car Dekho platformundan alınan araç detaylarını içeren `CAR DETAILS FROM CAR DEKHO.csv` adlı veri seti kullanılmıştır. Veri seti aşağıdaki sütunları içermektedir:

*   `name`: Araç modeli adı
*   `year`: Aracın üretim yılı
*   `selling_price`: Aracın satış fiyatı (hedef değişken)
*   `km_driven`: Aracın kat ettiği kilometre
*   `fuel`: Yakıt tipi (Benzin, Dizel, LPG, CNG, Elektrik)
*   `seller_type`: Satıcı türü (Bireysel, Bayi, Yetkili Bayi)
*   `transmission`: Vites kutusu tipi (Manuel, Otomatik)
*   `owner`: Araç sahibinin türü (Birinci Sahip, İkinci Sahip vb.)

## ⚙️ Uygulanan Yöntemler ve Analiz Adımları

Proje aşağıdaki adımları içermektedir:

### 1. Veri Keşfi ve Ön İşleme
*   Veri setinin genel yapısı (`.info()`, `.describe()`) incelenmiştir.
*   `year` sütunu kullanılarak aracın yaşı (`car_age`) hesaplanmış ve yeni bir özellik olarak eklenmiştir.
*   Sayısal özellikler arasındaki ilişkileri anlamak için korelasyon matrisi görselleştirilmiştir.

### 2. Basit Doğrusal Regresyon (Simple Linear Regression)
*   `car_age` özelliğini kullanarak `selling_price` tahmin etmeye yönelik basit bir doğrusal regresyon modeli oluşturulmuştur.
*   Modelin kesişim ve eğim değerleri ile R², MAE, RMSE gibi performans metrikleri hesaplanmıştır.

### 3. Çoklu Doğrusal Regresyon (Multiple Linear Regression) - Orijinal Veri ile
*   Kategorik değişkenler (`fuel`, `seller_type`, `transmission`, `owner`) `One-Hot Encoding` yöntemiyle sayısallaştırılmıştır.
*   `name` ve orijinal `year` sütunları modelden çıkarılarak diğer özelliklerle çoklu doğrusal regresyon modeli eğitilmiştir.
*   Modelin performans metrikleri (R², MAE, RMSE) değerlendirilmiş ve gerçek ile tahmin edilen değerler görselleştirilmiştir.

### 4. Aykırı Değer Tespiti ve Temizliği
*   Model performansını artırmak amacıyla `selling_price`, `km_driven` ve `car_age` sütunlarındaki aykırı değerler `Interquartile Range (IQR)` metodu kullanılarak tespit edilmiş ve temizlenmiştir.
*   Aykırı değer temizliği sonrası veri setinin yeni istatistiksel dağılımları incelenmiştir.

### 5. Çoklu Doğrusal Regresyon - Temizlenmiş Veri ile
*   Aykırı değerlerden arındırılmış ve kategorik değişkenleri sayısallaştırılmış veri seti üzerinde yeni bir çoklu doğrusal regresyon modeli eğitilmiştir.
*   Modelin performans metrikleri (R², MAE, RMSE) tekrar hesaplanmış ve orijinal modelin performansı ile karşılaştırılmıştır. Bu sayede aykırı değer temizliğinin etkisi gözlemlenmiştir.

### 6. Hata Dağılımı (Residuals) Analizi
*   Hem orijinal hem de temizlenmiş veriyle eğitilen modellerin hata dağılımları (residuals) histogram ve tahmin edilen değerlere karşı scatter plot ile görselleştirilmiştir.
*   Hata analizleri, modellerin varsayımlara ne kadar uyduğunu (normalite, homoscedasticity) değerlendirmek için kullanılmıştır.

### 7. Temizlenmiş Veri Seti İçin Korelasyon Matrisi
*   Aykırı değerlerden arındırılmış veri setindeki sayısal özellikler arasındaki korelasyonlar bir ısı haritası (`heatmap`) ile görselleştirilerek değişkenler arası ilişkiler yeniden incelenmiştir.

## 📈 Sonuçlar ve Model Performansı

*   **Basit Doğrusal Regresyon (Car Age ile):** Düşük R² skoru (yaklaşık 0.162) ile `car_age`'in tek başına satış fiyatını açıklamada yetersiz kaldığı görülmüştür.
*   **Orijinal Veri ile Çoklu Doğrusal Regresyon:** R² skoru yaklaşık 0.403 olarak elde edilmiştir. Hata analizi, heteroskedastisite ve normal dağılımdan sapmalar olduğunu göstermiştir.
*   **Temizlenmiş Veri ile Çoklu Doğrusal Regresyon:** Aykırı değer temizliği sayesinde model performansı önemli ölçüde iyileşmiştir. R² skoru yaklaşık 0.530'a yükselirken, MAE ve RMSE değerleri düşmüştür. Hata dağılımları daha normal ve homojen hale gelmiştir.

Bu sonuçlar, aykırı değerlerin model performansı üzerinde büyük bir etkisi olduğunu ve veri temizliğinin önemini vurgulamaktadır.

## 🛠️ Kurulum

Bu projeyi çalıştırmak için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
