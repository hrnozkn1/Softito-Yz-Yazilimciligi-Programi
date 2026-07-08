# 🚀 Telekom Müşteri Kaybı (Churn) Tahmin Modeli ve Optimizasyonu: XG vs AdaBoost

## Genel Bakış
Bu proje, bir telekom şirketindeki müşteri kaybını (churn) tahmin etmek amacıyla geliştirilmiştir. İlk olarak, AdaBoost ve XGBoost gibi popüler yükseltme (boosting) algoritmaları karşılaştırılmış, ardından en iyi performans gösteren modelin hiperparametreleri optimize edilerek doğruluk ve tahmin yeteneği artırılmıştır. Çalışma, veri ön işleme, model eğitimi, performans değerlendirmesi, özellik önem analizi ve ROC eğrisi çizimi gibi adımları içermektedir.

## 📋 İçerik

*   **1. Veri Yükleme ve Ön İşleme:** Müşteri verileri yüklenir, eksik değerler temizlenir ve kategorik özellikler sayısala dönüştürülerek modellemeye hazır hale getirilir.
*   **2. AdaBoost vs XGBoost Düellosu:** Başlangıç seviyesinde AdaBoost ve XGBoost modelleri eğitilerek performans metrikleri (Doğruluk, Duyarlılık, Kesinlik, F1-Skoru, AUC-ROC) karşılaştırılır.
*   **3. XGBoost Hiperparametre Optimizasyonu:** XGBoost modelinin performansını artırmak için `GridSearchCV` kullanılarak en iyi hiperparametre kombinasyonu (örneğin, `n_estimators`, `max_depth`, `learning_rate`) bulunur. Özellikle F1-Skoru'nun optimizasyon metriği olarak belirlenmesi, dengesiz veri setlerinde pozitif sınıf tahminini iyileştirmede önemli rol oynamıştır.
*   **4. Optimize Edilmiş XGBoost Modelinin Değerlendirilmesi:** Optimize edilmiş modelin test seti üzerindeki detaylı performans metrikleri sunulur ve başlangıçtaki modelle karşılaştırılır.
*   **5. XGBoost Modelinin Özellik Önem Derecesi (Feature Importance):** Modelin müşteri kaybını tahmin ederken hangi özelliklere daha fazla ağırlık verdiğini görselleştirilir. Bu analiz, churn'a yol açan temel faktörleri anlamada kritik rol oynar.
*   **6. ROC Eğrisi (Receiver Operating Characteristic Curve):** Modelin sınıflandırma performansını görsel olarak değerlendirmek için ROC eğrisi çizilir ve AUC değeri hesaplanır.

## 📈 Temel Bulgular ve Sonuçlar

*   **Model Seçimi:** İlk karşılaştırmada **XGBoost Classifier**, AdaBoost'a kıyasla daha iyi genel performans (Accuracy, F1-Score, AUC-ROC) göstermiştir.
*   **Hiperparametre Optimizasyonu:** `F1-Score` metriği kullanılarak yapılan optimizasyon, modelin özellikle `Recall` ve `F1-Score` değerlerini önemli ölçüde artırmıştır. En iyi parametreler: `{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 200}`.
*   **Performans İyileşmesi:** Optimize edilmiş XGBoost modeli, başlangıçtaki XGBoost modeline göre Accuracy'de 0.8800'den 0.8900'e, F1-Score'da 0.6786'dan 0.7080'e ve AUC-ROC'da 0.7825'ten 0.7921'e yükselerek önemli bir iyileşme sağlamıştır.
*   **Özellik Önemi:** `Contract_One_year`, `Contract_Two_year` ve `MonthlyCharges` gibi özellikler, müşteri kaybını tahmin etmede en etkili faktörler olarak belirlenmiştir. Bu, özellikle müşteri sözleşme tipleri ve aylık ücretlerin churn üzerindeki doğrudan etkisine işaret etmektedir.
*   **ROC Eğrisi ve AUC:** Optimize edilmiş modelin ROC eğrisi, 0.7921'lik bir AUC değeri ile oldukça iyi bir ayrıştırma gücüne sahip olduğunu göstermektedir.

## ✨ Öneriler

Modelin ortaya koyduğu özellik önem dereceleri doğrultusunda, müşteri kaybını azaltmaya yönelik aşağıdaki ticari stratejiler önerilebilir:

*   **Sözleşme Tipleri:** Uzun dönemli (bir veya iki yıllık) sözleşmelere geçişi teşvik etmek için özel teklifler ve indirimler sunulabilir. Kısa vadeli müşterilerle (aydan aya) daha proaktif iletişim kurulabilir.
*   **Aylık Ücretler:** Rekabetçi fiyatlandırma stratejileri gözden geçirilmeli ve müşterilere sunulan hizmetlerin değeri açıkça vurgulanmalıdır. Fiyat artışları öncesinde müşteri memnuniyetini artırıcı adımlar atılabilir.
*   **Hizmet Süresi (Tenure Months):** Yeni müşteriler için başarılı bir başlangıç süreci (onboarding) tasarlanmalı ve sadakat programları ile uzun süreli müşterilerin elde tutulması hedeflenmelidir.

Bu analizler, telekom şirketlerinin müşteri kaybını azaltmak ve müşteri sadakatini artırmak için daha bilinçli kararlar almasına yardımcı olacaktır.
