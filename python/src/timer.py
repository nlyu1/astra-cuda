import time 
from collections import deque

class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.elapsed_times = deque(maxlen=100)
        self.steps = deque(maxlen=100)

    def tick(self, num_steps):
        """
        Returns amortized steps per second
        """
        elapsed_time = time.time() - self.start_time
        self.start_time = time.time()
        self.elapsed_times.append(elapsed_time)
        self.steps.append(num_steps)
        return sum(self.steps) / sum(self.elapsed_times)
    
class OneTickTimer:
    def __init__(self):
        self.start_time = time.time()

    def tick(self):
        elapsed_time = time.time() - self.start_time
        self.start_time = time.time()
        return elapsed_time