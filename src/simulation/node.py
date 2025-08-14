from simulation.cache import LRUCache
from simulation.mec import MECServer

class EdgeNode:
    def __init__(self, node_id: int, cache_capacity: int, mec_service_rate: float):
        self.id = int(node_id)
        self.cache = LRUCache(cache_capacity)
        self.mec = MECServer(mec_service_rate)
        self.neighbor = None  # set by environment

    def reset(self):
        self.cache = LRUCache(self.cache.capacity)
        self.mec.reset()
