"""ETL Airflow Simülasyonu: DAG, Task, Operator, XCom"""
import time, random, os, logging, threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("airflow")

os.makedirs("cikti", exist_ok=True)

class XCom:
    """Task'lar arası veri paylaşım mekanizması"""
    def __init__(self):
        self.store: Dict[str, Any] = {}
    def push(self, key, value):
        self.store[key] = value
        logger.info(f"  XCom push: {key}")
    def pull(self, key):
        return self.store.get(key)

class Task:
    """DAG içindeki tek iş birimi"""
    def __init__(self, name: str, func: Callable, upstream: List[str] = None):
        self.name = name; self.func = func; self.upstream = upstream or []
        self.status = "pending"

    def run(self, xcom: XCom, context: dict):
        self.status = "running"
        logger.info(f"[{self.name}] başladı")
        try:
            result = self.func(xcom, context)
            xcom.push(self.name, result)
            self.status = "success"
            logger.info(f"[{self.name}] tamamlandı")
        except Exception as e:
            self.status = "failed"
            logger.error(f"[{self.name}] HATA: {e}")

class DAG:
    """Yönlendirilmiş Asiklik Graf"""
    def __init__(self, name: str, schedule: str = None):
        self.name = name; self.schedule = schedule; self.tasks: List[Task] = []

    def add_task(self, task: Task):
        self.tasks.append(task)

    def run(self):
        logger.info(f"DAG '{self.name}' başlatıldı")
        xcom = XCom()
        context = {"dag": self.name, "timestamp": datetime.now().isoformat()}
        completed = set()

        while len(completed) < len(self.tasks):
            for task in self.tasks:
                if task.name in completed: continue
                if all(u in completed for u in task.upstream):
                    task.run(xcom, context)
                    completed.add(task.name)
                    time.sleep(0.2)

        logger.info(f"DAG '{self.name}' tamamlandı")

# ── ETL Pipeline ───────────────────────────────────────────────────────
def extract(xcom, ctx):
    logger.info("  Veri çekiliyor (CSV simülasyonu)...")
    from faker import Faker
    import pandas as pd
    fake = Faker("tr_TR")
    data = [{"id": i, "isim": fake.name(), "sehir": fake.city(), "tutar": round(random.uniform(100, 5000), 2)}
            for i in range(1, 101)]
    df = pd.DataFrame(data)
    df.to_csv("cikti/ham_veri.csv", index=False)
    logger.info(f"  {len(data)} satır çekildi")
    return df

def transform(xcom, ctx):
    df = xcom.pull("extract")
    logger.info(f"  Dönüşüm yapılıyor ({len(df)} satır)...")
    df["tutar_tl"] = df["tutar"]
    df["tutar_usd"] = (df["tutar"] / 33).round(2)
    df["kategori"] = df["tutar"].apply(lambda x: "dusuk" if x < 1000 else ("orta" if x < 3000 else "yuksek"))
    df["islem_tarihi"] = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"  {len(df.columns)} sütun, 3 yeni eklendi")
    return df

def load(xcom, ctx):
    df = xcom.pull("transform")
    logger.info(f"  Veri hedefe yazılıyor...")
    df.to_csv("cikti/temiz_veri.csv", index=False)

    import sqlite3
    conn = sqlite3.connect("cikti/etl_db.sqlite")
    df.to_sql("satislar", conn, if_exists="replace", index=False)
    conn.close()
    logger.info(f"  SQLite'a yazıldı: cikti/etl_db.sqlite")
    return len(df)

def validate(xcom, ctx):
    import pandas as pd
    df = pd.read_csv("cikti/temiz_veri.csv")
    issues = []
    if df["tutar"].isnull().any(): issues.append("eksik tutar")
    if (df["tutar"] < 0).any(): issues.append("negatif tutar")
    if df.duplicated(subset=["id"]).any(): issues.append("mükerrer ID")
    status = "PASS" if not issues else f"FAIL: {', '.join(issues)}"
    logger.info(f"  Validasyon: {status}")
    return status

def notify(xcom, ctx):
    status = xcom.pull("validate")
    n = xcom.pull("load")
    logger.info(f"  Rapor: {n} satır işlendi, validasyon={status}")
    with open("cikti/etl_rapor.txt", "w") as f:
        f.write(f"ETL Raporu — {datetime.now()}\n")
        f.write(f"Satır: {n}, Durum: {status}\n")
    logger.info("  Rapor kaydedildi: cikti/etl_rapor.txt")

if __name__ == "__main__":
    print("ETL / Airflow Pipeline Simülasyonu\n")

    dag = DAG("etl_satis_pipeline", schedule="0 6 * * *")
    dag.add_task(Task("extract", extract))
    dag.add_task(Task("transform", transform, upstream=["extract"]))
    dag.add_task(Task("load", load, upstream=["transform"]))
    dag.add_task(Task("validate", validate, upstream=["load"]))
    dag.add_task(Task("notify", notify, upstream=["validate"]))

    dag.run()
    print("\nTamamlandı. Çıktılar cikti/ klasöründe.")
