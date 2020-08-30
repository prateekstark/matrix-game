import random
from RandomAgent import RandomAgent
from SimplifiedActionDecoderAgent import SimplifiedActionDecoderAgent
import torch


class Runner(object):
    def __init__(self, payoff_matrix, num_episodes):
        self.payoff_matrix = payoff_matrix
        self.agents = [SimplifiedActionDecoderAgent(0), SimplifiedActionDecoderAgent(1)]
        self.num_episodes = num_episodes
        self.moving_reward_average = 0

    def run(self):
        rewards = []
        for episode in range(num_episodes):
            epsilon = max(0.1, 1 - 2 * episode / num_episodes)
            episode_reward = 0
            history = []
            for agent in self.agents:
                agent.deal_card()
                move, greedy_move, q_value, greedy_q = agent.act(history, epsilon, True)
                history.append(
                    {
                        "exploratory_action": move,
                        "greedy_action": greedy_move,
                        "q_value": q_value,
                        "q_value_greedy": greedy_q,
                    }
                )

            episode_reward = self.payoff_matrix[
                history[0]["exploratory_action"] + (3 * self.agents[0].card)
            ][history[1]["exploratory_action"] + (3 * self.agents[1].card)]

            """
			We use a mean square loss here. The loss is calculated for the first agent as well as the second agent.
			"""
            for agent in self.agents:
                agent.optim.zero_grad()
            loss = torch.pow(episode_reward - history[0]["q_value"], 2) + torch.pow(
                episode_reward - history[1]["q_value"], 2
            )
            loss.backward()

            for agent in self.agents:
                agent.optim.step()

            self.moving_reward_average += 0.01 * (
                episode_reward - self.moving_reward_average
            )

            if episode % 1000 == 0:
                print(
                    "Episode: {}, Reward: {}".format(
                        episode, self.moving_reward_average
                    )
                )
            rewards.append(episode_reward)


if __name__ == "__main__":
    num_episodes = 1000000
    payoff_matrix = [
        [10, 0, 0, 0, 0, 10],
        [4, 8, 4, 4, 8, 4],
        [10, 0, 0, 0, 0, 10],
        [0, 0, 10, 10, 0, 0],
        [4, 8, 4, 4, 8, 4],
        [0, 0, 0, 10, 0, 0],
    ]
    print("Payoff Matrix: {}".format(payoff_matrix))
    runner = Runner(payoff_matrix, num_episodes)
    runner.run()
