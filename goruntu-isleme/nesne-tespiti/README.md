# Nesne Tespiti

YOLO ve SSD ile gerçek zamanlı nesne tespiti. İstanbul sokak sahnelerinde araç, insan, tabela bulma.

## Kapsam

- YOLOv8 nano model ile nesne tespiti
- SSD (Single Shot MultiBox Detector) ile karşılaştırma
- COCO128 veri seti ile eğitim
- Gerçek görseller üzerinde çıkarım
- Bounding box görselleştirme

## Kurulum

```bash
cd goruntu-isleme/nesne-tespiti
pip install -r requirements.txt
```

## Çalıştırma

```bash
python nesne_tespiti.py
```

İstanbul sokak fotoğrafı ile test etmek için: `python nesne_tespiti.py --resim fotograflar/istanbul_sokak.jpg`

## Kullanılanlar

`ultralytics` · `torchvision` · `OpenCV` · `matplotlib`
