import random


class RandomAgent(object):
    def __init__(self, rank):
        self.rank = rank

    def deal_card(self):
        self.card = random.choice([0, 1])

    def act(self, history):
        if self.rank == 0:
            assert len(history) == 0
        else:
            assert len(history) != 0

        return random.randint(0, 2)
