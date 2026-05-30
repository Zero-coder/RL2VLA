# Rebuttal Experiment Checklist

These are the experiments most directly aligned with the ACMMM 2026 reviews.

## Main ablation

| Method | ID | Visual OOD | Semantic OOD | Execution OOD |
|---|---:|---:|---:|---:|
| SFT/OpenVLA | TBD | TBD | TBD | TBD |
| PPO without EVI | TBD | TBD | TBD | TBD |
| PPO + FiLM-EVI | TBD | TBD | TBD | TBD |

## Simpler entropy-conditioning baselines

| Method | ID | Visual OOD | Semantic OOD | Execution OOD |
|---|---:|---:|---:|---:|
| PPO without EVI | TBD | TBD | TBD | TBD |
| PPO + entropy concat | TBD | TBD | TBD | TBD |
| PPO + entropy-weighted value loss | TBD | TBD | TBD | TBD |
| PPO + FiLM-EVI | TBD | TBD | TBD | TBD |

## EVI overhead

| Component | Extra parameters | Trainable-param ratio | Training memory | Inference latency |
|---|---:|---:|---:|---:|
| FiLM-EVI | TBD | TBD | TBD | TBD |

## Diagnostics

- Value calibration error by OOD split.
- TD error versus entropy bin.
- Return prediction error versus entropy bin.
- Advantage variance during PPO updates.
- Success-conditioned trajectory diversity.

## Wording guardrails

- Do not claim real-robot deployment robustness without real-robot experiments.
- Do not equate high entropy with failure. Use "policy uncertainty/action multimodality".
- Do not call the critic `Q(s,a)` unless the target and objective are changed.
- Report whether entropy descriptors are detached.
