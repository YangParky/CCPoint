import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, input_dim=2048, mid_dim=512, out_dim=256):
        super(Projector, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, mid_dim, bias=False),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, x):

        x = self.net(x)

        return x


class Predictor(nn.Module):
    def __init__(self, input_dim=2048, mid_dim=512, out_dim=256):
        super(Predictor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, mid_dim, bias=False),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, x):

        x = self.net(x)

        return x