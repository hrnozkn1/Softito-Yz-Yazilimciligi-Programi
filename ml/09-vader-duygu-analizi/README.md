# VADER Duygu Analizi: IMDB Film Yorumları

## Proje Hakkında

Bu proje, Natural Language Toolkit (NLTK) kütüphanesinin VADER (Valence Aware Dictionary and sEntiment Reasoner) aracını kullanarak IMDB film yorumlarının duygu analizini yapmaktadır. VADER, özellikle sosyal medya metinleri için optimize edilmiş, kural tabanlı bir duygu analiz modelidir. Bu çalışma, VADER'ın belirli bir veri seti üzerindeki performansını değerlendirmeyi, duygu skorlarının dağılımını görselleştirmeyi ve modelin güçlü/zayıf yönlerini analiz etmeyi amaçlamaktadır.

## Veri Seti

Analizde kullanılan veri seti, IMDB film yorumlarından oluşmaktadır ve `Train.csv` adlı dosyada bulunmaktadır. Her yorum için bir `text` (yorum metni) ve bir `label` (0: negatif, 1: pozitif) sütunu içermektedir. Veri seti, proje kapsamında `data/` dizini altına yerleştirilmiştir.

## Kurulum

Projeyi yerel olarak çalıştırmak için aşağıdaki adımları izleyin:

1.  Depoyu klonlayın:
    ```bash
    git clone https://github.com/KULLANICI_ADINIZ/PROJE_ADINIZ.git
    cd PROJE_ADINIZ
    ```
2.  Gerekli kütüphaneleri yükleyin (önerilen Python sürümü 3.x):
    ```bash
    pip install -r requirements.txt
    ```
    (`requirements.txt` dosyasını manuel olarak oluşturmanız gerekebilir. İçeriği `pandas`, `nltk`, `matplotlib`, `seaborn` olabilir.)
3.  NLTK VADER sözlüğünü indirin (Python ortamınızda bir kez çalıştırın):
    ```python
    import nltk
    nltk.download('vader_lexicon')
    ```
4.  `data/` dizini altına `Train.csv` dosyasını yerleştirin.
5.  `notebooks/vader_sentiment_analysis.ipynb` Jupyter Notebook dosyasını açarak analizleri inceleyebilir ve çalıştırabilirsiniz.

## Analiz ve Sonuçlar

### Duygu Analizi (VADER)

Her film yorumu için VADER kullanılarak bir `compound` (bileşik) duygu skoru hesaplanmıştır. Bu skor -1 (en negatif) ile +1 (en pozitif) arasında değişmektedir. Belirlenen eşik değerlerine göre (`compound >= 0.05` için pozitif, `compound <= -0.05` için negatif, aksi takdirde nötr) VADER tahminleri (`vader_prediction`) oluşturulmuştur.

### Temel Bulgular

*   **Doğruluk Oranı:** VADER algoritmasının bu veri seti üzerindeki doğruluk oranı yaklaşık **%69.33** olarak bulunmuştur.
*   **Skor Dağılımı:** VADER `compound` skorlarının dağılımı görselleştirilmiş, çoğu yorumun aşırı negatif veya aşırı pozitif skorlara sahip olduğu gözlemlenmiştir.
*   **Hatalı Tahmin Analizi:** Modelin yanlış tahmin ettiği yorumlar detaylı olarak incelenmiştir. Bu analizde, VADER'ın özellikle **sarkazm, ironi, ince mizah** veya **karmaşık cümle yapıları** içeren yorumlarda zorlandığı tespit edilmiştir. Örneğin, yorumun başında pozitif kelimeler kullanılıp devamında eleştirel bir dönüş yapıldığında veya tam tersi durumlarda modelin yanıldığı görülmüştür.

### Görselleştirmeler

Analiz sonuçları, `compound` skor dağılımını ve gerçek etiketlerle VADER tahminlerinin karşılaştırmasını gösteren grafiklerle desteklenmiştir.

### Sonuç

VADER, hızlı ve etkili bir duygu analiz aracıdır, özellikle ön işlem gerektirmeden iyi sonuçlar verebilir. Ancak, insan dilinin nüansları (sarkazm gibi) karşısında sınırlılıkları bulunmaktadır. Daha yüksek doğruluk oranları ve karmaşık duygu ifadelerini anlama yeteneği için daha gelişmiş makine öğrenimi veya derin öğrenme tabanlı modeller (`BERT`, `RoBERTa` vb.) düşünülmelidir.

## Katkıda Bulunma

Her türlü katkı ve öneriye açığız! Lütfen bir issue açın veya pull request gönderin.

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır.
