# Dağıtık Sistem Simülasyonu

HDFS, YARN ve Apache Spark'ın eğitim amaçlı mini simülasyonları. Thread'lerle dağıtık sistem mantığını gösterir.

## Kapsam

- **MINI HDFS**: NameNode, DataNode, block dağıtımı, replikasyon
- **MINI YARN**: ResourceManager, NodeManager, container tahsisi
- **MINI SPARK**: RDD, transformation (lazy), action (eager), DAG, WordCount

## Çalıştırma

```bash
cd BigData/dagik-sistem-simulasyonu
python main.py
```

Menüden istediğin sistemi seç: [1] HDFS, [2] YARN, [3] Spark, [4] Tümü.

## Kullanılanlar

`Python stdlib` · `threading`
