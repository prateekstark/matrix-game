import torch
import random
from agents.RandomAgent import RandomAgent
from agents.DeepCFR.DeepCFRAgent import DeepCFRAgent
from agents.DeepCFR.StrategyMemory import StrategyMemory
from copy import deepcopy
from statistics import mean


"""
Two randomization techniques we are using for avoiding local minima are:
    1) Restart the DeepCFR Advantage Networks from scratch after some iterations
    2) Pertubation in strategies which favours a single action. This could improve some exploration.
"""

"""
Things to notice:
    1) DeepCFR is quite sensitive to initializations!
"""


class CFRRunner(object):
    def __init__(
        self,
        payoff_matrix,
        num_iterations,
        K=100,
        num_runs=1000,
        advantage_batch_size=32,
        policy_batch_size=32,
        advantage_memory_size=20000,
        strategy_memory_size=20000,
    ):

        self.payoff_matrix = payoff_matrix
        self.player_list = [0, 1]
        self.advantage_memory_size = advantage_memory_size
        self.strategy_memory_size = strategy_memory_size
        self.agents = [
            DeepCFRAgent(0, advantage_memory_size),
            DeepCFRAgent(1, advantage_memory_size),
        ]
        self.num_iterations = num_iterations
        self.K = K
        self.strategy_memory = StrategyMemory(
            self.player_list, self.strategy_memory_size
        )
        self.advantage_batch_size = advantage_batch_size
        self.policy_batch_size = policy_batch_size
        self.num_runs = num_runs

    def run(self):
        for i in range(self.num_runs):
            if i % 100 == 0:
                self.set_advantage_weights_zero()

            self.DeepCFR(i * self.num_iterations)

            print()
            print("Game Strength: {}".format(self.evaluate()))
            print()

    def DeepCFR(self, start_iter):

        """
        Initialize each player's advantage network, such that it returns 0 for all outputs.
        Initialize reservoir sampled memories for advantages and a strategy memory.
        """

        for iteration in range(self.num_iterations):
            """
            For training from scratch in each iteration
            """
            for agent in self.agents:
                agent.deal_card()

            for player in self.player_list:
                for traversal in range(self.K):
                    self.TRAVERSE(
                        [], self.strategy_memory, (iteration + 1 + start_iter), player
                    )
                self.agents[player].sample_advantage_and_train(
                    self.advantage_batch_size
                )

        """
        Train for policy network of each agent to converge to the strategy.
        """

        for policy_iteration in range(100):
            for player in self.player_list:
                data = self.strategy_memory.sample(player, self.policy_batch_size)
                self.agents[player].train_policy_net(data, self.policy_batch_size)
            if policy_iteration % 10 == 0:
                print(
                    "Partial Game Strength in iteration {}: {}".format(
                        policy_iteration, self.evaluate()
                    )
                )

    def TRAVERSE(self, history, strategy_memory, t, p):
        """
        if h is terminal then return the payoff to player p
        """
        if len(history) == 2:
            player1 = history[0]["action"] + torch.Tensor([3 * self.agents[0].card])
            player2 = history[1]["action"] + torch.Tensor([3 * self.agents[1].card])
            return_val = self.payoff_matrix[int(player1)][int(player2)]

            return return_val

        """
        if h is a chance node, then sample an action from the probabilities. But in our two player matrix game, there is no chance node, so we do not need to cansider them!
        """

        """
        if P(h) = p, then compute strategy from predicted advantages using regret matching
        """

        if len(history) == p:
            """
            Compute strategy, from advantage network
            """
            I = self.agents[p].get_infoset(history)
            advantage_values = self.agents[p].compute_advantage(I)
            advantage_values = torch.clamp(advantage_values, min=0)

            strategy = (
                advantage_values / advantage_values.sum()
                if advantage_values.sum() > 0.000001
                else torch.full_like(advantage_values, 1 / advantage_values.shape[0])
            )

            v = torch.zeros(3)
            expected_v = 0
            for action in [0, 1, 2]:
                pseudo_history = deepcopy(history)
                pseudo_history.append({"action": action})
                v[action] = self.TRAVERSE(pseudo_history, strategy_memory, t, p)
                with torch.no_grad():
                    expected_v += strategy[action] * v[action]

            """
            Compute Advantages
            """

            regrets = torch.zeros(3)
            for action in [0, 1, 2]:
                regrets[action] = v[action] - expected_v

            """
            Insert the infoset and action advantages into the advantage memory
            """

            buffer_object = {"infoset": I, "t": t, "regrets": regrets}
            self.agents[p].advantage_memory.add_memory_object(buffer_object)

        else:
            I = self.agents[1 - p].get_infoset(history)
            advantage_values = self.agents[1 - p].compute_advantage(I)
            advantage_values = torch.clamp(advantage_values, min=0)
            strategy = (
                advantage_values / advantage_values.sum()
                if advantage_values.sum() > 0.000001
                else torch.full_like(advantage_values, 1 / advantage_values.shape[0])
            )

            """
            Insert infoset and action probabilities into strategy memory.
            """

            strategy_memory_object = {"infoset": I, "t": t, "strategy": strategy}
            self.strategy_memory.insert_element(1 - p, strategy_memory_object)

            """
            Sample an action a from probability distribution of the strategy
            """

            action = random.choices([0, 1, 2], weights=strategy, k=1)[0]
            history.append({"action": action})

            return self.TRAVERSE(history, strategy_memory, t, p)

    def evaluate(self):
        rewards = []
        for _ in range(1000):
            history = []
            for agent in self.agents:
                agent.deal_card()
                move = agent.act_policy(history)
                history.append(
                    {
                        "action": move,
                    }
                )

            player1 = history[0]["action"] + 3 * self.agents[0].card

            player2 = history[1]["action"] + 3 * self.agents[1].card

            episode_reward = self.payoff_matrix[int(player1)][int(player2)]
            rewards.append(episode_reward)

        return mean(rewards)

    def add_strategy_pertubation1(self, strategy):
        index = (strategy == 1).nonzero()
        if index.shape[0] == 0:
            return strategy

        assert index.shape == (1, 1)
        index = index.item()

        torch.fill_(strategy, 0.15 / strategy.shape[0])

        strategy[index] += 0.85

        return strategy

    def add_strategy_pertubation2(self, strategy):
        strategy += 0.2
        return strategy / strategy.sum()

    def set_advantage_weights_zero(self):
        for agent in self.agents:
            agent.advantage_net.init_weights()
