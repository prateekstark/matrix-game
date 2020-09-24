import random


class StrategyMemory(object):
    def __init__(self, player_list, max_size=10000):
        self.buffers = {}
        self.max_size = max_size
        for player in player_list:
            self.buffers[player] = []

    def insert_element(self, player, element):
        self.buffers[player].append(element)
        if len(self.buffers[player]) > self.max_size:
            self.buffers[player].pop(random.randrange(len(self.buffers[player])))

    def sample(self, player, batch_size):
        return random.choices(self.buffers[player], k=batch_size)

    def get_size(self, player):
        return len(self.buffers[player])
