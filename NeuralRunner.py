import torch
import random
from agents.RandomAgent import RandomAgent
from agents.SimplifiedActionDecoderAgent import SimplifiedActionDecoderAgent


class NeuralRunner(object):
    def __init__(
        self, payoff_matrix, num_episodes, num_runs, epsilon_limit=0.05, batch_size=32
    ):
        self.payoff_matrix = payoff_matrix
        self.batch_size = batch_size
        self.agents = [
            SimplifiedActionDecoderAgent(0, batch_size),
            SimplifiedActionDecoderAgent(1, batch_size),
        ]
        self.num_episodes = num_episodes
        self.epsilon_limit = epsilon_limit
        self.num_runs = num_runs

    def run(self):
        for _ in range(self.num_runs):
            rewards = []
            for episode in range(self.num_episodes):
                epsilon = max(self.epsilon_limit, 1 - 2 * episode / self.num_episodes)
                history = []

                for agent in self.agents:
                    agent.deal_cards(self.batch_size)
                    move, greedy_move, q_value, greedy_q = agent.act(
                        history, epsilon, True
                    )
                    history.append(
                        {
                            "exploratory_action": move,
                            "greedy_action": greedy_move,
                            "q_value": q_value,
                            "q_value_greedy": greedy_q,
                        }
                    )
                player1 = history[0]["exploratory_action"] + torch.Tensor(
                    [3 * card for card in self.agents[0].cards]
                )
                player2 = history[1]["exploratory_action"] + torch.Tensor(
                    [3 * card for card in self.agents[1].cards]
                )
                episode_reward = torch.zeros(self.batch_size)

                for index in range(self.batch_size):
                    episode_reward[index] = self.payoff_matrix[
                        int(player1[index].item())
                    ][int(player2[index].item())]

                """
    			We use a mean square loss here. The loss is calculated for the first agent as well as the second agent.
    			"""

                for agent in self.agents:
                    agent.optim.zero_grad()
                loss = (
                    torch.pow(episode_reward - history[0]["q_value"], 2)
                    + torch.pow(episode_reward - history[1]["q_value"], 2)
                ).norm()
                loss.backward()

                for agent in self.agents:
                    agent.optim.step()

                if episode % 10 == 0:
                    print(
                        "Episode: {}, Reward: {}".format(episode, episode_reward.mean())
                    )
                rewards.append(episode_reward)

    def evaluate(self):
        history = []
        for agent in self.agents:
            agent.deal_cards(self.batch_size)
            move, greedy_move, q_value, greedy_q = agent.act(history, 0, True)
            history.append(
                {
                    "exploratory_action": move,
                    "greedy_action": move,
                    "q_value": q_value,
                    "q_value_greedy": greedy_q,
                }
            )

        player1 = history[0]["exploratory_action"] + torch.Tensor(
            [3 * card for card in self.agents[0].cards]
        )

        player2 = history[1]["exploratory_action"] + torch.Tensor(
            [3 * card for card in self.agents[1].cards]
        )

        episode_reward = torch.zeros(self.batch_size)

        for index in range(self.batch_size):
            episode_reward[index] = self.payoff_matrix[int(player1[index].item())][
                int(player2[index].item())
            ]

        return episode_reward.mean()
