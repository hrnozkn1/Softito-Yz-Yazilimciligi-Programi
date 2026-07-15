# ETL / Airflow Pipeline

Apache Airflow mimarisinin Python simülasyonu. DAG, Task, Operator, XCom kavramlarını thread'lerle gösterir.

## Kapsam

- DAG (Directed Acyclic Graph) yapısı
- Task ve Operator kavramları
- XCom ile task'lar arası veri paylaşımı
- Schedule ve trigger mekanizmaları
- ETL akışı: Extract → Transform → Load → Validate → Notify

## Çalıştırma

```bash
cd BigData/etl-airflow-pipeline
pip install -r requirements.txt
python etl_airflow.py
```

## Kullanılanlar

`pandas` · `Faker` · `Python stdlib`
