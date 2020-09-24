import random
import torch
from utils import to_one_hot, to_one_hot_multiple
from agents.DeepCFR.models.PolicyNet import PolicyNet
from agents.DeepCFR.models.AdvantageNet import AdvantageNet
from agents.DeepCFR.AdvantageMemory import AdvantageMemory

"""
To Do: So, I'm planning to 
    1) make the code more modular
    2) I need to figure out what's wrong by putting print statements everywhere
    3) Need to check the hyperparameters
    4) If it still doesn't help then think over the theory again!
    5) Should we try adding pertubation?
    6) Should we add LinearCFR?
"""


class DeepCFRAgent(object):
    def __init__(self, rank, advantage_memory_size):

        self.rank = rank
        self.advantage_net = self.init_advantage_weights()
        self.advantage_optim = torch.optim.Adam(
            self.advantage_net.parameters(), lr=0.001
        )
        self.policy_net = self.init_policy_weights()
        self.policy_optim = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.advantage_memory = AdvantageMemory(advantage_memory_size)

    def deal_card(self):
        self.card = random.choice([0, 1])

    def compute_advantage(self, infoset):
        with torch.no_grad():
            return self.advantage_net(infoset)

    def sample_advantage_and_train(self, batch_size):
        if self.advantage_memory.get_memory_size() > batch_size:
            for _ in range(10):
                data = self.advantage_memory.sample(batch_size)
                r = []
                I = []
                t = []

                for buffer_object in data:
                    regrets = buffer_object["regrets"]
                    timestep = buffer_object["t"]
                    infoset = buffer_object["infoset"]
                    r.append(regrets)
                    I.append(infoset)
                    t.append(timestep)

                I = torch.stack(I)
                r = torch.stack(r)
                t = torch.Tensor(t)

                weight_sum = t.sum()

                weights = t / weight_sum

                self.advantage_optim.zero_grad()

                loss = (
                    weights * (torch.pow(r - self.advantage_net(I), 2)).sum(-1)
                ).sum()
                loss /= batch_size
                loss.backward()

                """
                Gradient norm clipping to 1.
                """

                torch.nn.utils.clip_grad_norm_(self.advantage_net.parameters(), 1)

                self.advantage_optim.step()

    def train_policy_net(self, data, batch_size):

        strategies = []
        I = []
        t = []

        for buffer_object in data:
            strategy = buffer_object["strategy"]
            timestep = buffer_object["t"]
            infoset = buffer_object["infoset"]
            strategies.append(strategy)
            I.append(infoset)
            t.append(timestep)

        strategies = torch.stack(strategies)
        I = torch.stack(I)
        t = torch.Tensor(t)
        weight_sum = t.sum()

        weights = t / weight_sum

        self.policy_optim.zero_grad()

        loss = (weights * (torch.pow(strategies - self.policy_net(I), 2)).sum(-1)).sum()
        loss /= batch_size
        loss.backward(retain_graph=True)

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)

        self.policy_optim.step()

    def get_infoset(self, history):
        if self.rank == 0:
            assert len(history) == 0
            return to_one_hot(2, self.card)
        else:
            assert len(history) == 1
            return to_one_hot(6, (3 * self.card) + history[0]["action"])

    def act(self, history):
        infoset = self.get_infoset(history)
        advantage_values = self.compute_advantage(infoset)
        advantage_values = torch.clamp(advantage_values, min=0)

        strategy = (
            advantage_values / advantage_values.sum()
            if advantage_values.sum() > 0.000001
            else torch.full_like(advantage_values, 1 / advantage_values.shape[0])
        )
        action = random.choices([0, 1, 2], weights=strategy, k=1)[0]

        return action

    def act_policy(self, history):
        infoset = self.get_infoset(history)

        with torch.no_grad():
            policy = self.policy_net(infoset)

        action = random.choices([0, 1, 2], weights=policy, k=1)[0]
        return action

    def init_advantage_weights(self):
        if self.rank == 0:
            """
            Total Possible Cards: 2 in matrix game
            """
            input_dim = 2
            output_dim = 3
        else:
            """
            Input Space -> Number of Cards * Previous Action
            Advantage Network
            """
            input_dim = 2 * 3
            output_dim = 3

        return AdvantageNet(input_dim, output_dim)

    def init_policy_weights(self):
        if self.rank == 0:
            """
            Total Possible Cards: 2 in matrix game
            """
            input_dim = 2
            output_dim = 3
        else:
            """
            Input Space -> Number of Cards * Previous Action
            Policy Network
            """
            input_dim = 2 * 3
            output_dim = 3

        return PolicyNet(input_dim, output_dim)
