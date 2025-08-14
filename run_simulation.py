import sys, os, csv, random
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from simulation.environment import HybridCDNEnvironment
from simulation.config import CONFIG

os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)

env = HybridCDNEnvironment(CONFIG)

# Train Q-learning
episodes = CONFIG["TRAINING_EPISODES"]
rewards = []
for ep in range(1, episodes+1):
    env.agent.set_epsilon(max(CONFIG["EPSILON_END"], CONFIG["EPSILON_DECAY"] * env.agent.epsilon))
    r = env.run_episode(training=True)
    rewards.append(r)
    print(f"Episode {ep}/{episodes} | avg reward={r:.3f} | epsilon={env.agent.epsilon:.2f}")

with open("data/q_learning_convergence.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["Episode","AverageReward"])
    for i, r in enumerate(rewards, 1): w.writerow([i, f"{r:.6f}"])

plt.figure()
plt.plot(range(1, episodes+1), rewards, marker='o')
plt.title("Q-learning Convergence")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.grid(True)
plt.savefig("plots/q_learning_convergence.png")
plt.close()

def run_eval(env, enable_cache, enable_mec, use_rl, duration=20.0):
    coop = CONFIG["COOPERATIVE_CACHING"]
    CONFIG["COOPERATIVE_CACHING"] = bool(use_rl)
    env = HybridCDNEnvironment(CONFIG)
    env.agent.set_epsilon(0.0)
    total_latency = 0.0
    total_reqs = 0
    content_reqs = 0
    total_hits = 0
    env.reset()
    current_time = 0.0
    next_arrival = [random.expovariate(CONFIG["REQ_RATE_PER_CELL"]) for _ in range(CONFIG["NUM_CELLS"])]
    while current_time < duration:
        cell = int(np.argmin(next_arrival))
        current_time = next_arrival[cell]
        if current_time >= duration: break
        node = env.nodes[cell]
        is_content = (random.random() < CONFIG["CONTENT_REQUEST_RATIO"])
        if is_content:
            content_reqs += 1
            cid = env.sample_content()
            if enable_cache and node.cache.contains(cid):
                node.cache.get(cid)
                total_hits += 1
                latency = CONFIG["EDGE_PROP_DELAY"]*2 + CONFIG["TX_DELAY"]
            else:
                if enable_cache and use_rl and CONFIG.get("COOPERATIVE_CACHING", False):
                    state = ('content', 'hit' if node.cache.contains(cid) else 'miss')
                    act = env.agent.choose_action(state, [0,1,2])
                    latency = env._route_request(node, 'content', cid, act, current_time)
                else:
                    latency = CONFIG["CLOUD_PROP_DELAY"]*2 + CONFIG["TX_DELAY"]
                    if enable_cache: node.cache.put(cid)
        else:
            if enable_mec:
                if use_rl:
                    state = ('compute', 'idle' if node.mec.is_idle(current_time) else 'busy')
                    act = env.agent.choose_action(state, [0,1])
                    latency = env._route_request(node, 'compute', None, act, current_time)
                else:
                    finish, wait, service = node.mec.process_task(current_time)
                    latency = CONFIG["EDGE_PROP_DELAY"]*2 + (wait*1000.0) + (service*1000.0)
            else:
                service = random.expovariate(CONFIG["MEC_SERVICE_RATE"])
                latency = CONFIG["CLOUD_PROP_DELAY"]*2 + (service*1000.0)
        total_latency += latency; total_reqs += 1
        next_arrival[cell] += random.expovariate(CONFIG["REQ_RATE_PER_CELL"])
    CONFIG["COOPERATIVE_CACHING"] = coop
    hit_rate = (total_hits / content_reqs) if content_reqs else 0.0
    avg_latency = (total_latency / total_reqs) if total_reqs else 0.0
    return hit_rate, avg_latency

scenarios = {
    "Traditional":   (False, False, False),
    "EdgeOnly":      (True,  False, False),
    "MECOnly":       (False, True,  False),
    "HybridGreedy":  (True,  True,  False),
    "HybridRL":      (True,  True,  True),
}

results = {}
for name, (c,m,r) in scenarios.items():
    hr, lat = run_eval(env, c, m, r, duration=CONFIG["EPISODE_DURATION"])
    results[name] = (hr, lat)
    print(f"{name:13s} | HitRate={hr*100:6.2f}% | AvgLatency={lat:6.2f} ms")

# CSVs
import csv
with open("data/cache_hit_comparison.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["Scenario","CacheHitRatio"])
    for k in ["Traditional","EdgeOnly","MECOnly","HybridRL"]:
        w.writerow([k, f"{results[k][0]:.4f}"])

with open("data/latency_comparison.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["Scenario","AvgLatency(ms)"])
    for k in ["Traditional","EdgeOnly","MECOnly","HybridRL"]:
        w.writerow([k, f"{results[k][1]:.2f}"])

with open("data/latency_ablation.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["Scenario","AvgLatency(ms)"])
    for k in ["EdgeOnly","MECOnly","HybridGreedy","HybridRL"]:
        w.writerow([k, f"{results[k][1]:.2f}"])

# Plots (no explicit colors to keep defaults)
labels = ["Traditional","EdgeOnly","MECOnly","HybridRL"]
import matplotlib.pyplot as plt
plt.figure(); plt.bar(labels, [results[k][0]*100 for k in labels])
plt.ylabel("Cache Hit Rate (%)"); plt.title("Cache Hit Ratio Comparison"); plt.tight_layout()
plt.savefig("plots/cache_hit_comparison.png"); plt.close()

plt.figure(); plt.bar(labels, [results[k][1] for k in labels])
plt.ylabel("Avg End-to-End Latency (ms)"); plt.title("End-to-End Latency Comparison"); plt.tight_layout()
plt.savefig("plots/latency_comparison.png"); plt.close()

# Sensitivity: Zipf α
alphas = [0.6, 0.8, 1.0, 1.2]
sens = []
for a in alphas:
    from simulation.config import CONFIG as C
    C["ZIPF_EXPONENT"] = a
    env_a = HybridCDNEnvironment(C)
    # brief retrain for new dist
    env_a.agent.set_epsilon(1.0)
    for _ in range(10): env_a.run_episode(training=True)
    env_a.agent.set_epsilon(0.0)
    # Edge-only vs HybridRL
    def eval_pair(envx):
        # Edge-only
        hr_e, _ = run_eval(envx, True, False, False, duration=C["EPISODE_DURATION"])
        # HybridRL
        hr_h, _ = run_eval(envx, True, True, True, duration=C["EPISODE_DURATION"])
        return hr_e, hr_h
    e, h = eval_pair(env_a)
    sens.append((a, e, h))

with open("data/zipf_sensitivity.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["ZipfAlpha","EdgeOnly_HitRate","HybridRL_HitRate"])
    for a, e, h in sens:
        w.writerow([a, f"{e:.4f}", f"{h:.4f}"])

plt.figure()
plt.plot([x[0] for x in sens], [x[1]*100 for x in sens], 'o--', label="EdgeOnly")
plt.plot([x[0] for x in sens], [x[2]*100 for x in sens], 's--', label="HybridRL")
plt.xlabel("Zipf Exponent (α)"); plt.ylabel("Cache Hit Rate (%)"); plt.title("Hit Rate vs Content Skew"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("plots/zipf_sensitivity.png"); plt.close()

print("Done. CSVs in ./data, figures in ./plots")
