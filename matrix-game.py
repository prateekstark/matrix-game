from Runner import Runner
from NeuralRunner import NeuralRunner
from CFRRunner import CFRRunner
import torch

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    num_iterations = 1000
    payoff_matrix = [
        [10, 0, 0, 0, 0, 10],
        [4, 8, 4, 4, 8, 4],
        [10, 0, 0, 0, 0, 10],
        [0, 0, 10, 10, 0, 0],
        [4, 8, 4, 4, 8, 4],
        [0, 0, 0, 10, 0, 0],
    ]

    print("Payoff Matrix: {}".format(payoff_matrix))
    runner = CFRRunner(payoff_matrix, num_iterations=100, num_runs=1000, K=500)
    runner.run()
