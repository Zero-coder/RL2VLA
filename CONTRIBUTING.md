# Contributing

Thanks for your interest in RL2VLA.

## Development setup

```bash
pip install -e ".[dev]"
pytest -q
```

## Code style

- Keep method code independent from heavyweight benchmark imports where possible.
- Do not import OpenVLA, ManiSkill, or SimplerEnv at package import time.
- Add unit tests for lightweight modules such as entropy computation, FiLM-EVI, PPO/GAE, and metrics.

## Experiment reporting

When adding results, report:

- benchmark split and task IDs
- number of evaluation episodes
- random seeds
- mean and standard deviation
- checkpoint path and config file
- hardware and runtime

## Pull requests

Please keep pull requests scoped. If a change touches benchmark adapters and algorithm logic, split it when possible.
