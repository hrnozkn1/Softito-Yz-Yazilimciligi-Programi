"""Dağıtık Sistem Simülasyonu: HDFS + YARN + Spark (eğitim amaçlı thread simülasyonu)"""
import threading, uuid, time, random
from collections import defaultdict

class Colors:
    H, B, C, G, W, F, E = '\033[95m','\033[1m','\033[96m','\033[92m','\033[93m','\033[91m','\033[0m'

# ── MINI HDFS ────────────────────────────────────────────────────────────────
class DataNode:
    def __init__(self, node_id):
        self.id = node_id; self.blocks = {}
    def store(self, block_id, data):
        self.blocks[block_id] = data
        print(f"    [{self.id}] block {block_id} kaydedildi ({len(data)} bayt)")
    def get(self, block_id):
        return self.blocks.get(block_id)

class NameNode:
    def __init__(self, replication=2):
        self.files = {}
        self.data_nodes = [DataNode(f"DN-{i}") for i in range(3)]
        self.replication = replication
    def write(self, filename, data, block_size=32):
        blocks = [data[i:i+block_size] for i in range(0, len(data), block_size)]
        block_ids = [str(uuid.uuid4())[:8] for _ in blocks]
        self.files[filename] = block_ids
        print(f"{Colors.C}  HDFS: '{filename}' → {len(blocks)} block (replikasyon={self.replication}){Colors.E}")
        for bid, bdata in zip(block_ids, blocks):
            nodes = random.sample(self.data_nodes, min(self.replication, len(self.data_nodes)))
            for node in nodes:
                node.store(bid, bdata)
    def read(self, filename):
        if filename not in self.files:
            return f"{Colors.F}Dosya bulunamadı!{Colors.E}"
        data = ""
        for bid in self.files[filename]:
            for node in self.data_nodes:
                block = node.get(bid)
                if block:
                    data += block
                    break
        return data

def hdfs_demo():
    print(f"\n{Colors.H}═══ MINI HDFS ═══{Colors.E}")
    nn = NameNode(replication=2)
    nn.write("musteriler.txt", "ID,ISIM,SEHIR\n1,Ahmet,Istanbul\n2,Ayse,Ankara\n3,Mehmet,Izmir")
    result = nn.read("musteriler.txt")
    print(f"  Okunan: {result[:60]}...")
    print(f"{Colors.G}  HDFS demo tamamlandi.{Colors.E}")

# ── MINI YARN ────────────────────────────────────────────────────────────────
class Container:
    def __init__(self, cid, cpu, ram):
        self.id = cid; self.cpu = cpu; self.ram = ram; self.running = True
    def run(self, task):
        print(f"    Container({self.id}) isleniyor: {task}...")
        time.sleep(0.3)
        print(f"    Container({self.id}) tamamlandi: {task}")
        self.running = False

class NodeManager:
    def __init__(self, nm_id, cpu_total, ram_total):
        self.id = nm_id; self.cpu_total = cpu_total; self.ram_total = ram_total
        self.cpu_used = 0; self.ram_used = 0; self.containers = []
    def allocate(self, task, cpu, ram):
        if self.cpu_used + cpu <= self.cpu_total and self.ram_used + ram <= self.ram_total:
            cid = f"container-{len(self.containers)}"
            c = Container(cid, cpu, ram)
            self.containers.append(c)
            self.cpu_used += cpu; self.ram_used += ram
            t = threading.Thread(target=c.run, args=(task,))
            t.start()
            return True
        return False

class ResourceManager:
    def __init__(self):
        self.nodes = [NodeManager(f"NM-{i}", cpu_total=4, ram_total=8192) for i in range(3)]
    def submit(self, app_name, tasks):
        print(f"  RM: '{app_name}' alindi ({len(tasks)} gorev)")
        for task in tasks:
            placed = False
            for nm in self.nodes:
                if nm.allocate(task, cpu=1, ram=1024):
                    placed = True; break
            if not placed:
                print(f"  {Colors.F}Kaynak yok: {task} bekletiliyor...{Colors.E}")
        time.sleep(2)

def yarn_demo():
    print(f"\n{Colors.H}═══ MINI YARN ═══{Colors.E}")
    rm = ResourceManager()
    rm.submit("WordCount", ["map-partition-0", "map-partition-1", "reduce-final"])
    print(f"{Colors.G}  YARN demo tamamlandi.{Colors.E}")

# ── MINI SPARK ────────────────────────────────────────────────────────────────
class RDD:
    def __init__(self, data, deps=None, func=None):
        self.data = list(data)
        self.partitions = self._split(data, 3) if not deps else self._compute(data, func)
    def _split(self, data, n):
        chunk = max(1, len(data) // n)
        return [data[i:i+chunk] for i in range(0, len(data), chunk)]
    def _compute(self, parent, func):
        return [func(p) for p in parent.partitions]
    def map(self, func):
        return RDD([], deps=self, func=lambda p: [func(x) for x in p])
    def filter(self, func):
        return RDD([], deps=self, func=lambda p: [x for x in p if func(x)])
    def flatMap(self, func):
        return RDD([], deps=self, func=lambda p: [item for x in p for item in func(x)])
    def reduceByKey(self, func):
        groups = defaultdict(list)
        for p in self.collect():
            k, v = p
            groups[k].append(v)
        return dict((k, sum(v)) for k, v in groups.items())
    def collect(self):
        result = []
        for p in self.partitions:
            result.extend(p)
        return result
    def count(self):
        return len(self.collect())

def spark_demo():
    print(f"\n{Colors.H}═══ MINI SPARK ═══{Colors.E}")
    rdd = RDD(["merhaba dünya", "merhaba spark", "dünya büyük", "spark hizli"])
    print(f"  RDD oluşturuldu: {rdd.count()} satir")

    kelimeler = rdd.flatMap(lambda x: x.split())
    print(f"  flatMap sonrasi kelimeler: {kelimeler.collect()}")

    filtrelenmis = kelimeler.filter(lambda x: len(x) > 4)
    print(f"  filter(len>4): {filtrelenmis.collect()}")

    wordcount = rdd.flatMap(lambda x: x.split()).map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b)
    print(f"  WordCount sonucu: {dict(sorted(wordcount.items()))}")
    print(f"{Colors.G}  Spark demo tamamlandi.{Colors.E}")

# ── MENU ──────────────────────────────────────────────────────────────────────
def menu():
    while True:
        print(f"\n{Colors.B}  [1] HDFS  [2] YARN  [3] Spark  [4] Tümü  [0] Çıkış{Colors.E}")
        c = input("Seçim: ").strip()
        if c == "1": hdfs_demo()
        elif c == "2": yarn_demo()
        elif c == "3": spark_demo()
        elif c == "4": hdfs_demo(); yarn_demo(); spark_demo()
        elif c == "0": print("Çıkıldı."); break

if __name__ == "__main__":
    import os
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
        except: pass
    menu()
