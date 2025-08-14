# Hybrid CDN Simulation (Edge Cache + MEC + Q-learning)

This repository provides a **working simulation** for a hybrid CDN that combines edge caching, MEC offloading, and a tabular Q-learning controller.
It reproduces the core figures from the paper using synthetic traffic (Zipf popularity + Poisson arrivals).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_simulation.py
```

Outputs (CSVs under `data/`, PNGs under `plots/`):
- `q_learning_convergence.csv/png`
- `cache_hit_comparison.csv/png`
- `latency_comparison.csv/png`
- `latency_ablation.csv/png`
- `zipf_sensitivity.csv/png`

## Project layout

```
HybridCDN-Sim/
├── run_simulation.py
├── requirements.txt
├── LICENSE
├── README.md
├── src/
│   ├── simulation/
│   │   ├── config.py
│   │   ├── agent.py
│   │   ├── cache.py
│   │   ├── mec.py
│   │   ├── node.py
│   │   └── environment.py
│   └── utils/
│       └── logger.py (placeholder)
├── data/   # generated CSVs
└── plots/  # generated PNGs
```

## Configuration

Edit `src/simulation/config.py` to change: number of cells, cache size, Zipf α, request rate, MEC service rate, RL hyperparameters, and cooperative caching.

## Reproducibility

All randomness is seeded (`RAND_SEED` in config). Figures are generated deterministically with that seed. For confidence intervals, run multiple seeds and average CSVs.

## License

MIT — see `LICENSE`.
