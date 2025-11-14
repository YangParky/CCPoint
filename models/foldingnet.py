import itertools
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class FoldingNet(nn.Module):
    """ FoldingNet.
    Used in many methods, e.g. FoldingNet, PCN, OcCo, Point-BERT
    learning point reconstruction only from global feature
    """
    def __init__(self, input_dim=1024, grid_size=4, grid_scale=0.05):
        super().__init__()

        self.grid_size = grid_size
        self.grid_scale = grid_scale
        self.num_coarse = 1024
        self.num_fine = self.grid_size ** 2 * self.num_coarse  # 16384

        self.meshgrid = [[-self.grid_scale, self.grid_scale, self.grid_size],
                         [-self.grid_scale, self.grid_scale, self.grid_size]]

        self.folding1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_coarse * 3))

        self.folding2 = nn.Sequential(
            nn.Conv1d(input_dim+2+3, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_grid(self, batch_size):
        x, y = np.linspace(*self.meshgrid[0]), np.linspace(*self.meshgrid[1])
        points = np.array(list(itertools.product(x, y)))
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)

        return torch.tensor(points).float().to(self.device)

    def tile(self, tensor, multiples):
        def tile_single_axis(a, dim, n_tile):
            init_dim = a.size()[dim]
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*repeat_idx)
            order_index = torch.Tensor(
                np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).long()
            return torch.index_select(a, dim, order_index.to(self.device))

        for dim, n_tile in enumerate(multiples):
            if n_tile == 1:
                continue
            tensor = tile_single_axis(tensor, dim, n_tile)

        return tensor

    @staticmethod
    def expand_dims(tensor, dim):

        return tensor.unsqueeze(-1).transpose(-1, dim)

    def forward(self, x):
        coarse = self.folding1(x)
        coarse = coarse.view(-1, self.num_coarse, 3)

        grid = self.build_grid(x.shape[0])
        grid_feat = grid.repeat(1, self.num_coarse, 1)

        point_feat = self.tile(self.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        point_feat = point_feat.view([-1, self.num_fine, 3])

        global_feat = self.tile(self.expand_dims(x, 1), [1, self.num_fine, 1])
        feat = torch.cat([grid_feat, point_feat, global_feat], dim=2)

        center = self.tile(self.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        center = center.view([-1, self.num_fine, 3])

        fine = self.folding2(feat.transpose(2, 1)).transpose(2, 1) + center

        return coarse, fine