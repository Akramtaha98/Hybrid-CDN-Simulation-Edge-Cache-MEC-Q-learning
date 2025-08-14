CONFIG = {
    "NUM_CELLS": 10,
    "CONTENT_CATALOG_SIZE": 1000,
    "ZIPF_EXPONENT": 0.8,
    "REQ_RATE_PER_CELL": 5.0,        # requests/sec per cell
    "CONTENT_REQUEST_RATIO": 0.7,    # 70% content, 30% compute

    "CACHE_SIZE": 100,               # items
    "MEC_SERVICE_RATE": 50.0,        # tasks/sec (μ)

    # Delays (milliseconds)
    "EDGE_PROP_DELAY": 5.0,
    "NEIGHBOR_PROP_DELAY": 10.0,
    "CLOUD_PROP_DELAY": 50.0,
    "TX_DELAY": 0.5,

    # RL hyperparameters
    "TRAINING_EPISODES": 50,
    "EPISODE_DURATION": 20.0,        # seconds simulated per episode
    "LEARNING_RATE": 0.1,
    "DISCOUNT_FACTOR": 0.9,
    "EPSILON_START": 1.0,
    "EPSILON_END": 0.1,
    "EPSILON_DECAY": 0.95,           # multiplicative per episode

    "COOPERATIVE_CACHING": True,     # neighbor lookups for content
    "RAND_SEED": 42,
}
