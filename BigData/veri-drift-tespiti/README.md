# Veri Drift Tespiti

Veri dağılımındaki değişimi KS-test, Jensen-Shannon Divergence ve Wasserstein Distance ile tespit etme.

## Kapsam

- Kolmogorov-Smirnov testi (dağılım karşılaştırma)
- Jensen-Shannon Divergence (olasılık uzaklığı)
- Wasserstein Distance (Earth Mover's Distance)
- 4 farklı drift senaryosu

## Çalıştırma

```bash
cd BigData/veri-drift-tespiti
pip install -r requirements.txt
python veri_drift.py
```

## Kullanılanlar

`NumPy` · `SciPy` · `matplotlib`
