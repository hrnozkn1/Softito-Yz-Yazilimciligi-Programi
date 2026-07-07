# Softito_Yz_Yazilimciligi_Programi

## NLP Projeleri

| # | Proje | Durum |
|---|-------|-------|
| 01 | [tf-idf](nlp/01-tf-idf/) — TF-IDF: Teoriden Pratiğe | ✅ Tamamlandı |
| 02 | [word-embeddings](nlp/02-word-embeddings/) — Word2Vec, FastText, t-SNE | ✅ Tamamlandı |
| 03 | [rnn](nlp/03-rnn/) — Vanilla RNN (TTC-4900) | ✅ Tamamlandı |
| 04 | [lstm](nlp/04-lstm/) — LSTM (AG News) | ✅ Tamamlandı |
| 05 | [attention](nlp/05-attention/) — LSTM + Attention (AG News) | ✅ Tamamlandı |
| 06 | [transformer](nlp/06-transformer/) — Transformer Encoder (AG News) | ✅ Tamamlandı |

Detaylı bilgi için: [`nlp/README.md`](nlp/README.md)

| # | Veri | Model | Accuracy |
|---|------|-------|----------|
| 01 | TTC-4900 (Türkçe, 7 sınıf) | TF-IDF + LogisticRegression | — |
| 02 | TTC-4900 (Türkçe, 7 sınıf) | Word2Vec / FastText / TF-IDF | — |
| 03 | TTC-4900 (Türkçe, 7 sınıf) | Vanilla RNN (PyTorch) | %41 |
| 04 | AG News (İngilizce, 4 sınıf, 120K) | LSTM | — |
| 05 | AG News (İngilizce, 4 sınıf, 120K) | BiLSTM + Attention | — |
| 06 | AG News (İngilizce, 4 sınıf, 120K) | Transformer Encoder | — |

> 03-rnn sonuçları test edildi ve push edildi. 04-05-06 scriptleri hazır, eğitim tam sonuç için kullanıcının çalıştırması gerekir (~15-20dk CPU).

```bash
git clone https://github.com/hrnozkn1/Softito-Yz-Yazilimciligi-Programi.git
cd Softito-Yz-Yazilimciligi-Programi/nlp/04-lstm
pip install -r requirements.txt
python lstm_siniflandirma.py
```