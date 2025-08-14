import time 
from collections import defaultdict
from collections import deque

class Timer:
    def __init__(self, length=11):
        self.start_time = time.time()
        self.elapsed_times = deque(maxlen=length)
        self.steps = deque(maxlen=length)

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
    
class SegmentTimer:
    def __init__(self):
        self.elapsed_times = {}
        self.last_name = None 
        self.last_time = None 

    def tick(self, name):
        if self.last_name is None:
            self.last_name = name 
            self.last_time = time.time() * 1e3
            return 
        new_time = time.time() * 1e3
        self.elapsed_times[self.last_name] = new_time - self.last_time
        self.last_time = new_time
        self.last_name = name 