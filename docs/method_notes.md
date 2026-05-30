# Method Notes

## Critic notation

The EVI critic should be written as:

```text
V_phi(s_t; u_t)
```

where `u_t` is a policy-derived uncertainty descriptor from the current rollout. This avoids the misleading impression that EVI is a pure state-only `V(s)` while also avoiding the incorrect interpretation that it is a Q-learning `Q(s,a)` target.

## Entropy descriptor

For tokenized actions, compute per-token categorical entropy and aggregate:

```text
H_k = - sum_i p_{k,i} log p_{k,i}
H = mean_k H_k
```

The default implementation normalizes each token entropy by `log(vocab_size)`.

## Gradient flow

The default config sets `detach_descriptor: true`. This means the critic loss does not update actor logits through the entropy descriptor path. PPO actor updates still come from the clipped policy objective.

## FiLM-EVI

EVI maps entropy to FiLM parameters:

```text
gamma, beta = MLP(H)
h_v' = gamma * h_v + beta
V = linear(h_v')
```

The final FiLM layer is initialized to identity modulation (`gamma=1`, `beta=0`) so PPO starts from the unmodulated critic.
