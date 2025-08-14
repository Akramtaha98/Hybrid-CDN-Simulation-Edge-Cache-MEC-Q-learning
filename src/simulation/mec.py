import random

class MECServer:
    def __init__(self, service_rate: float):
        self.service_rate = float(service_rate)  # μ tasks/sec
        self.next_free_time = 0.0
        self.utilization_time = 0.0
        self.last_idle_check = 0.0

    def process_task(self, current_time: float):
        # utilization bookkeeping
        if self.next_free_time > self.last_idle_check:
            busy_until = min(current_time, self.next_free_time)
            if busy_until > self.last_idle_check:
                self.utilization_time += (busy_until - self.last_idle_check)
        self.last_idle_check = current_time

        start_service = max(current_time, self.next_free_time)
        wait = start_service - current_time
        service = random.expovariate(self.service_rate)  # seconds
        finish = start_service + service
        self.next_free_time = finish
        return finish, wait, service

    def is_idle(self, current_time: float) -> bool:
        return current_time >= self.next_free_time

    def reset(self):
        self.next_free_time = 0.0
        self.utilization_time = 0.0
        self.last_idle_check = 0.0
