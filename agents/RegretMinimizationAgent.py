import random
import torch
from utils import to_one_hot, to_one_hot_multiple
import torch.nn as nn
import torch.nn.functional as F


class Memory(object):
    def __init__(self):
        self.buffer = []


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RegretMinimizationAgent(object):
    def __init__(self, rank, batch_size):
        self.rank = rank
        self.weights = self.init_weights()
        self.optim = torch.optim.SGD([self.weights], lr=0.001)
        self.batch_size = batch_size

    def deal_cards(self, batch_size):
        self.cards = random.choices([0, 1], k=batch_size)

    def act(self, history, epsilon, train):
        if self.rank == 0:
            assert len(history) == 0
            net_input = to_one_hot_multiple(2, self.cards)
        else:
            assert len(history) != 0
            net_input = to_one_hot_multiple(
                18,
                torch.Tensor([9 * element for element in self.cards])
                + history[0]["exploratory_action"] * 3
                + history[0]["greedy_action"] * int(train),
            )

        net_input = net_input.float()

        """
        Size of Neural Network Output: batch_size * num_actions
        """

        logits = torch.matmul(net_input, self.weights)
        best_action = torch.argmax(logits, dim=1)

        probabilities = epsilon / 3 + (1 - epsilon) * to_one_hot_multiple(
            3, best_action
        )
        log_probs = torch.log(probabilities)

        exploratory_actions = torch.zeros(self.batch_size)
        for index in range(self.batch_size):
            distribution = torch.distributions.multinomial.Multinomial(
                logits=log_probs[index]
            )
            exploratory_action = torch.argmax(distribution.sample())
            exploratory_actions[index] = exploratory_action

        greedy_actions = torch.argmax(log_probs, dim=1)

        q_value = torch.zeros(self.batch_size)
        q_value_greedy = torch.zeros(self.batch_size)

        for index in range(self.batch_size):
            q_value[index] = logits[index][exploratory_actions[index].long()]
            q_value_greedy[index] = logits[index][greedy_actions[index].long()]

        return exploratory_actions, greedy_actions, q_value, q_value_greedy

    def init_weights(self):
        if self.rank == 0:
            """
            Total Possible Cards: 2 in matrix game
            """
            input_dim = 2
            output_dim = 3
        else:
            """
            Joint Action Q-value
            Number of Cards * Greedy Action * Actual Action
            """
            input_dim = 2 * 3 * 3
            output_dim = 3

        return torch.nn.init.normal_(
            torch.zeros(input_dim, output_dim, requires_grad=True), mean=0, std=1
        )
