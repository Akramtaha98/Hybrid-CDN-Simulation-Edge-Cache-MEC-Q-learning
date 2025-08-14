from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.storage = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0

    def contains(self, content_id) -> bool:
        return content_id in self.storage

    def get(self, content_id) -> bool:
        if content_id in self.storage:
            self.storage.move_to_end(content_id)
            self.hit_count += 1
            return True
        else:
            self.miss_count += 1
            return False

    def put(self, content_id):
        if content_id in self.storage:
            self.storage.move_to_end(content_id)
            return
        if len(self.storage) >= self.capacity:
            self.storage.popitem(last=False)  # evict LRU
        self.storage[content_id] = None
