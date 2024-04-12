import torch
import torch.nn as nn
import torch.nn.functional as F
from util import Model

class get_model(nn.Module):
    def __init__(self,num_class):
        super(get_model, self).__init__()
        self.model = Model(points=10000, embed_dim=64, groups=1, res_expansion=1.0,
                           activation="relu", bias=False, use_xy=False, normalize=None,
                           dim_expansion=[1, 1], pre_blocks=[2, 2], pos_blocks=[2, 2],
                           radiuses=[0.05, 0.08], k_neighbors=[100, 100], reducers=[2, 2])

        # Motion LSTM
        self.lstm = nn.LSTM(64,64, 1, dropout=0.5)

        self.classifier = nn.Sequential(
            nn.Linear(64,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_class))
        )
w
    def forward(self, xy):
        B, T, N, C = xy.shape              # [B,T,N,C]
        xy = xy.view(-1, 1000, 2)          # [B*T,N,C]
        x = self.model(xy)                 # [B*T,D]
        x = x.view(B,T,64)                 # [B,T,D]
        x = x.permute([1, 0, 2])           # [T,B,D]
        x, _ = self.lstm(x)                # [T,B,D]
        x = torch.mean(x, dim=0)           # [B,D]
        x =  self.classifier(x)
        x = F.log_softmax(x, -1)
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss


