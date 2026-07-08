# Machine Learning Projeleri

Makine öğrenmesi alanında denetimli, denetimsiz öğrenme, zaman serisi, doğal dil işleme ve keşifçi veri analizi konularını kapsayan 9 proje.

## Projeler

| # | Proje | Kategori | Algoritmalar |
|---|-------|----------|-------------|
| 01 | [Futbolcu Kümeleme](01-futbolcu-kumeleme/) | Denetimsiz Öğrenme | K-Means, GMM, Hiyerarşik Kümeleme, PCA |
| 02 | [Hava Durumu RNN](02-hava-durumu-rnn/) | Zaman Serisi | RNN (PyTorch) |
| 03 | [Kalp Hastalığı Tahmini](03-kalp-hastaligi-tahmin/) | Sınıflandırma | Logistic Regression, SVM |
| 04 | [Araba Fiyat Regresyonu](04-araba-fiyat-regresyon/) | Regresyon | Linear Regression |
| 05 | [Spotify Churn EDA](05-spotify-churn-eda/) | Keşifçi Veri Analizi | Pandas, Seaborn |
| 06 | [Su Kalitesi Sınıflandırması](06-su-kalitesi-siniflandirma/) | Sınıflandırma | KNN, Gaussian Naive Bayes |
| 07 | [Telefon Fiyat Sınıflandırması](07-telefon-fiyat-siniflandirma/) | Sınıflandırma | Decision Tree, Random Forest |
| 08 | [Telekom Churn XGBoost](08-telekom-churn-xgboost/) | Sınıflandırma | AdaBoost, XGBoost |
| 09 | [VADER Duygu Analizi](09-vader-duygu-analizi/) | Doğal Dil İşleme | VADER (NLTK) |

## Kullanılan Teknolojiler

- Python 3.10+
- PyTorch, scikit-learn, pandas, numpy
- Matplotlib, Seaborn (görselleştirme)
- NLTK (doğal dil işleme)
- Jupyter Notebook

## Çalıştırma

Her proje bağımsız bir Jupyter Notebook'tur. Çalıştırmak için:

```bash
cd Softito-Yz-Yazilimciligi-Programi/ml/XX-proje-adi
pip install -r requirements.txt   # veya pip install pandas numpy matplotlib seaborn scikit-learn torch nltk
jupyter notebook
```

Notebook'u açıp hücreleri sırayla çalıştırın. Veri setleri her projenin kendi klasöründe bulunur.
