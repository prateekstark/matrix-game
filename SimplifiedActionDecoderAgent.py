import random
import torch
from utils import to_one_hot


class SimplifiedActionDecoderAgent(object):
    def __init__(self, rank):
        self.rank = rank
        self.weights = self.init_weights()
        self.optim = torch.optim.Adam([self.weights])

    def deal_card(self):
        self.card = random.choice([0, 1])

    def act(self, history, epsilon, train):
        if self.rank == 0:
            assert len(history) == 0
            net_input = to_one_hot(2, self.card)
        else:
            assert len(history) != 0
            net_input = to_one_hot(
                18,
                self.card * 9
                + history[0]["exploratory_action"] * 3
                + history[0]["greedy_action"] * int(train),
            )

        net_input = net_input.float()
        logits = torch.matmul(net_input, self.weights)

        best_action = torch.argmax(logits)

        probabilities = epsilon / 3 + (1 - epsilon) * to_one_hot(3, best_action)
        log_probs = torch.log(probabilities)

        distribution = torch.distributions.multinomial.Multinomial(logits=log_probs)
        exploratory_action = torch.argmax(distribution.sample())
        greedy_action = torch.argmax(log_probs)

        q_value = logits[exploratory_action]
        q_value_greedy = logits[greedy_action]

        return exploratory_action, greedy_action, q_value, q_value_greedy

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
