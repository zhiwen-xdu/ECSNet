import torch
import torch.nn as nn
import torch.nn.functional as F
from util import Model

class get_model(nn.Module):
    def __init__(self,num_class):
        super(get_model, self).__init__()
        self.model = Model(points=5000, embed_dim=128, groups=1, res_expansion=1.0,
                           activation="relu", bias=False, use_xy=False, normalize="anchor",
                           dim_expansion=[1, 1], pre_blocks=[2, 2], pos_blocks=[2, 2],
                           radiuses=[0.035,0.055], k_neighbors=[128,128], reducers=[2, 2])

        self.classifier = nn.Sequential(
            nn.Linear(128,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_class))

    def forward(self, xy):
        B, N, C = xy.shape          # [B,N,C]
        x = self.model(xy)          # [B,D]
        x =  self.classifier(x)
        x = F.log_softmax(x, -1)
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss


