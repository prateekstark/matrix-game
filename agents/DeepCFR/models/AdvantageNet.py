import torch.nn as nn
import torch.nn.functional as F

class AdvantageNet(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, 6)
        self.fc4 = nn.Linear(6, output_dim)
        self.init_weights()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def init_weights(self):
        nn.init.normal_(self.fc1.weight)
        nn.init.normal_(self.fc2.weight)
        nn.init.normal_(self.fc3.weight)
        nn.init.normal_(self.fc4.weight)
        nn.init.normal_(self.fc1.bias)
        nn.init.normal_(self.fc2.bias)
        nn.init.normal_(self.fc3.bias)
        nn.init.normal_(self.fc4.bias)

    def init_zero_weights(self):
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc4.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)

if __name__ == '__main__':
    import torch
    net = AdvantageNet(6)
    print(net)
    random_input = torch.rand(6)
    print("NN Output: {}".format(net(random_input)))