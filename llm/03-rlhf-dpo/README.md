# 03 - RLHF + DPO

Model hizalama: Reward model, PPO ve DPO.

## Kapsam

- Tercih verisiyle reward model eğitimi
- PPO ile RLHF (reinforcement learning from human feedback)
- DPO (direct preference optimization)
- Base vs RLHF vs DPO karşılaştırması

## Kurulum

```bash
cd llm/03-rlhf-dpo
pip install -r requirements.txt
```

## Çalıştırma

```bash
python rlhf_dpo.py
```

## Kullanılanlar

`Transformers` · `trl` · `PEFT` · `datasets` · `accelerate`
