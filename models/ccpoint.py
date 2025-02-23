import numpy as np
import torch
import torch.nn as nn
import itertools

from timm.models.layers import trunc_normal_

from models.mlp import Projector, Predictor
from models.foldingnet import FoldingNet


class CCPoint(nn.Module):
    def __init__(self, args, backbone):
        super(CCPoint, self).__init__()

        self.encoder = backbone
        self.proj1 = Projector(input_dim=args.emb_dims, mid_dim=2048, out_dim=256)
        self.prid1 = Predictor(input_dim=256, mid_dim=768, out_dim=256)

        self.proj2 = Projector(input_dim=args.emb_dims, mid_dim=2048, out_dim=256)
        self.foldingnet = FoldingNet(input_dim=1024)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, point1, point2=None, corrupt_point=None):

        if corrupt_point is not None:
            point_proj1 = self.proj1(self.encoder(point1))
            point_prid1 = self.prid1(point_proj1)
            point_proj2 = self.proj1(self.encoder(point2))
            point_prid2 = self.prid1(point_proj2)

            corrup_feat = self.encoder(corrupt_point)
            corrup_proj = self.proj2(corrup_feat)

            corrup_coarse, corrup_fine = self.foldingnet(corrup_feat)

            return point_proj1, point_prid1, point_proj2, point_prid2, corrup_proj, corrup_coarse, corrup_fine
        else:
            point_feat = self.encoder(point1)

            return point_feat, point_feat