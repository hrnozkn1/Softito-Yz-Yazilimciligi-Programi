# 📊 Spotify Customer Churn - Kesifsel Veri Analizi (EDA)

Bu proje, Spotify kullanıcılarının platformu terk etme (churn) eğilimlerini anlamak ve veri setindeki dinamikleri incelemek amacıyla gerçekleştirilmiş bir **Keşifsel Veri Analizi (EDA)** çalışmasıdır.

## 📁 Proje Yapısı

* `Spotify_Customer_Churn_Exploratory_Data_Analysis_(EDA).ipynb`: Veri analizi, görselleştirme ve istatistiksel çıkarımların yer aldığı ana Jupyter Notebook dosyası.
* `datasets/spotify_churn_dataset.csv`: Analizde kullanılan, 8000 satır ve 12 özellikten oluşan Spotify kullanıcı verisi.

## 📊 Veri Seti Hakkında

Analiz edilen veri setinde yer alan temel kullanıcı özellikleri ve metrikler şunlardır:
* `user_id`: Kullanıcı kimliği
* `gender` & `age` & `country`: Demografik bilgiler
* `subscription_type`: Üyelik türü (Free, Premium, Student, Family)
* `listening_time`: Toplam dinleme süresi
* `songs_played_per_day`: Günlük oynatılan şarkı sayısı
* `skip_rate`: Şarkı geçme oranı
* `device_type`: Kullanılan cihaz türü (Mobile, Desktop, Web)
* `ads_listened_per_week`: Haftalık maruz kalınan reklam sayısı
* `offline_listening`: Çevrimdışı dinleme özelliği kullanımı (0 / 1)
* `is_churned`: Kullanıcının ayrılıp ayrılmadığı bilgisi (**Hedef Değişken - 0 / 1**)

## 🛠️ Kullanılan Teknolojiler

Projede veri işleme, istatistiksel analiz ve görselleştirme için aşağıdaki Python kütüphaneleri kullanılmıştır:
* **Veri Yönetimi:** `Pandas`, `NumPy`
* **Veri Görselleştirme:** `Matplotlib`, `Seaborn`
* **İstatistik:** `SciPy (stats)`

## 🚀 Kurulum ve Çalıştırma

Projeyi yerel bilgisayarınızda çalıştırmak için:

1. Bu repoyu klonlayın:
   ```bash
   git clone [https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git](https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git)
