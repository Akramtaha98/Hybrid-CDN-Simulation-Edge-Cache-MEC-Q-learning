import random, numpy as np
from simulation.node import EdgeNode
from simulation.agent import QLearningAgent

class HybridCDNEnvironment:
    def __init__(self, config):
        self.cfg = dict(config)
        self.nodes = [EdgeNode(i, config["CACHE_SIZE"], config["MEC_SERVICE_RATE"])
                      for i in range(config["NUM_CELLS"])]
        # neighbor ring if cooperation enabled
        if config.get("COOPERATIVE_CACHING", True):
            n = config["NUM_CELLS"]
            for i in range(n):
                self.nodes[i].neighbor = self.nodes[(i+1) % n]
        # Agent
        self.agent = QLearningAgent(config["LEARNING_RATE"],
                                    config["DISCOUNT_FACTOR"],
                                    config["EPSILON_START"])
        # Zipf popularity
        N = config["CONTENT_CATALOG_SIZE"]
        alpha = config["ZIPF_EXPONENT"]
        ranks = np.arange(1, N+1)
        weights = ranks ** (-alpha)
        self.popularity = (weights / weights.sum())
        # Seeds
        random.seed(config["RAND_SEED"])
        np.random.seed(config["RAND_SEED"])

    def reset(self):
        for node in self.nodes:
            node.reset()

    def sample_content(self):
        return int(np.random.choice(self.cfg["CONTENT_CATALOG_SIZE"], p=self.popularity))

    def run_episode(self, training=True):
        self.reset()
        duration = float(self.cfg["EPISODE_DURATION"])
        current_time = 0.0
        next_arrival = [random.expovariate(self.cfg["REQ_RATE_PER_CELL"])
                        for _ in range(self.cfg["NUM_CELLS"])]
        total_reward, nreq = 0.0, 0
        while current_time < duration:
            cell = int(np.argmin(next_arrival))
            current_time = next_arrival[cell]
            if current_time >= duration:
                break
            node = self.nodes[cell]
            is_content = (random.random() < self.cfg["CONTENT_REQUEST_RATIO"])
            if is_content:
                cid = self.sample_content()
                state_key = 'hit' if node.cache.contains(cid) else 'miss'
                state = ('content', state_key)
                actions = [0,1,2]  # Edge, Neighbor, Cloud
            else:
                cid = None
                state_key = 'idle' if node.mec.is_idle(current_time) else 'busy'
                state = ('compute', state_key)
                actions = [0,1]    # MEC, Cloud

            act = self.agent.choose_action(state, actions)
            latency_ms = self._route_request(node, 'content' if is_content else 'compute', cid, act, current_time)
            reward = max(0.0, 1.0 - (latency_ms / 100.0))
            total_reward += reward; nreq += 1
            if training:
                self.agent.update(state, act, reward, state)
            next_arrival[cell] += random.expovariate(self.cfg["REQ_RATE_PER_CELL"])
        return (total_reward / nreq) if nreq else 0.0

    def _route_request(self, node, rtype, cid, act, t):
        ms = 0.0
        if rtype == 'content':
            # 0 Edge, 1 Neighbor, 2 Cloud
            if act == 0:
                if node.cache.contains(cid):
                    node.cache.get(cid)
                    ms = self.cfg["EDGE_PROP_DELAY"] * 2 + self.cfg["TX_DELAY"]
                else:
                    ms = self.cfg["EDGE_PROP_DELAY"] * 2
                    ms += self.cfg["CLOUD_PROP_DELAY"] * 2 + self.cfg["TX_DELAY"]
                    node.cache.put(cid)
            elif act == 1:
                nb = node.neighbor
                if nb and nb.cache.contains(cid):
                    nb.cache.get(cid)
                    ms = self.cfg["NEIGHBOR_PROP_DELAY"] * 2 + self.cfg["TX_DELAY"]
                    node.cache.put(cid)
                else:
                    ms = self.cfg["NEIGHBOR_PROP_DELAY"] * 2
                    ms += self.cfg["CLOUD_PROP_DELAY"] * 2 + self.cfg["TX_DELAY"]
                    node.cache.put(cid)
            else:
                ms = self.cfg["CLOUD_PROP_DELAY"] * 2 + self.cfg["TX_DELAY"]
                node.cache.put(cid)
            return ms
        else:
            # compute: 0 MEC, 1 Cloud
            if act == 0:
                finish, wait, service = node.mec.process_task(t)
                ms = self.cfg["EDGE_PROP_DELAY"] * 2 + (wait*1000.0) + (service*1000.0)
            else:
                service = random.expovariate(node.mec.service_rate)
                ms = self.cfg["CLOUD_PROP_DELAY"] * 2 + (service*1000.0)
            return ms
