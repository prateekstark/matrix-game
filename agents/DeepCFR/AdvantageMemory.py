import random


class AdvantageMemory(object):
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size

    def sample(self, batch_size):
        return random.choices(self.buffer, k=batch_size)

    def add_memory_object(self, x):
        self.buffer.append(x)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(random.randrange(len(self.buffer)))

    def get_memory_size(self):
        return len(self.buffer)
