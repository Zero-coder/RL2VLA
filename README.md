# RL2VLA

Official implementation scaffold for **RL2VLA: Reinforcement Learning with Entropy-Aware Value Injection for Vision-Language-Action Generalization**.

RL2VLA improves Vision-Language-Action (VLA) policy robustness by adding reinforcement-learning post-training after supervised fine-tuning. The main method component is **Entropy-aware Value Injection (EVI)**, a critic-side FiLM modulation module that conditions value estimation on token-level action entropy.

> Status: initial open-source release scaffold. Training adapters are designed for OpenVLA-style tokenized action policies and ManiSkill/SimplerEnv-style manipulation benchmarks. Fill in dataset paths, pretrained checkpoints, and benchmark-specific environment adapters before large-scale runs.

## Highlights

- SFT-to-RL training recipe for VLA post-training.
- Entropy-aware Value Injection (EVI) critic module.
- PPO/GAE utilities for tokenized action policies.
- Config-driven experiment launcher.
- Reproducibility hooks for ablations requested by reviewers:
  - PPO without EVI
  - PPO + entropy concatenation
  - PPO + entropy-weighted value loss
  - PPO + FiLM-EVI

## Repository Layout

```text
RL2VLA/
  configs/
    default.yaml
    ablations/
      ppo_no_evi.yaml
      entropy_concat.yaml
      value_loss_weighting.yaml
      film_evi.yaml
  docs/
    rebuttal_experiments.md
    method_notes.md
  rl2vla/
    algorithms/
      ppo.py
    models/
      evi.py
      value_heads.py
    policy/
      openvla_adapter.py
    envs/
      maniskill_adapter.py
    utils/
      entropy.py
      seed.py
      metrics.py
  scripts/
    train_ppo.py
    eval_policy.py
    measure_overhead.py
  tests/
    test_entropy.py
    test_evi.py
```

## Installation

```bash
conda create -n rl2vla python=3.10 -y
conda activate rl2vla

pip install -e ".[dev]"
```

For full OpenVLA/ManiSkill training, install the corresponding upstream dependencies:

```bash
# Example only. Match CUDA/PyTorch versions to your machine.
pip install torch torchvision torchaudio
pip install mani-skill
pip install transformers accelerate peft
```

## Quick Smoke Test

The unit tests only validate the lightweight RL2VLA modules and do not require OpenVLA weights.

```bash
pytest -q
```

## Training

### PPO with FiLM-EVI

```bash
python scripts/train_ppo.py --config configs/ablations/film_evi.yaml
```

### PPO without EVI

```bash
python scripts/train_ppo.py --config configs/ablations/ppo_no_evi.yaml
```

### Entropy concat ablation

```bash
python scripts/train_ppo.py --config configs/ablations/entropy_concat.yaml
```

### Entropy-weighted value-loss ablation

```bash
python scripts/train_ppo.py --config configs/ablations/value_loss_weighting.yaml
```

## Method Summary

For a tokenized action sequence, RL2VLA computes token-level categorical entropy:

```text
H_k = - sum_i p_{k,i} log p_{k,i}
```

The entropy descriptor is aggregated across action tokens and normalized. EVI maps this descriptor to FiLM parameters:

```text
gamma, beta = MLP(H)
h_value' = gamma * h_value + beta
V = value_head(h_value')
```

The resulting critic is best viewed as a policy-conditioned PPO baseline:

```text
V_phi(s_t; u_t)
```

where `u_t` is a rollout-derived uncertainty descriptor. It is not a Q-learning action-value target; GAE and the PPO clipped actor objective remain standard.

## Reviewer-Focused Experiments

See [docs/rebuttal_experiments.md](docs/rebuttal_experiments.md) for the exact tables to fill:

- SFT/OpenVLA vs PPO vs PPO+EVI across ID, visual OOD, semantic OOD, and execution OOD.
- FiLM-EVI vs entropy concat vs entropy-weighted value loss.
- EVI overhead: parameters, memory, latency.
- Entropy-return diagnostics.

## Citation

```bibtex
@inproceedings{jiang2026rl2vla,
  title={RL2VLA: Reinforcement Learning with Entropy-Aware Value Injection for Vision-Language-Action Generalization},
  author={Jiang, Maowei and Wang, Qi and Ai, Hongfeng and Zeng, Pengyu and Liu, Ruikai and Li, Ruiqi and Wang, Yifan and Wang, Zihang and Yue, Sun and Liu, Quangao and Bus, Peter and Hu, Yusong and Dongfang, Yang and Liang, Alan and Miao, Rui and Shen, Zehao and Cheng, Moquan and Dong, Zhiyong},
  booktitle={ACM Multimedia},
  year={2026}
}
```

## Acknowledgements

This repository is designed for OpenVLA-style policies and ManiSkill/SimplerEnv-style embodied manipulation benchmarks. The repository structure is inspired by recent open-source VLA-RL projects while implementing the RL2VLA/EVI method independently.

## License

MIT License. See [LICENSE](LICENSE).
