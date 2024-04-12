import torch
import torch.nn as nn
import torch.nn.functional as F
from event_sample import farthest_point_sample, random_sample, surface_event_sample  



def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points



def query_ball(radius, nsample, xy, new_xy):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xy: all points, [B, N, 2]
        new_xy: query points, [B, S, 2]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xy.device
    B, N, C = xy.shape
    _, S, _ = new_xy.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xy, xy)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_event(nsample, xy, new_xy):
    """
    Input:
        nsample: max sample number in local region
        xy: all points, [B, N, 2]
        new_xy: query points, [B, S, 2]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xy, xy)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx



#   ==== Sampling + Grouping + Normalization ====
class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, radius, kneighbors, use_xy=True, normalize="center", **kwargs):
        """
        Give xy[B,N,2] and fea[B,N,D], return new_xy[B,S,2] and new_fea[B,S,K,D]
        :groups: cetriods                 S
        :radius: grouping, radius
        :kneighbors: k-nerighbors         K
        :use_xy: whether to concat xy
        :kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.radius = radius
        self.kneighbors = kneighbors
        self.use_xy = use_xy

        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel=2 if self.use_xy else 0
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1,1,1, channel + add_channel]))

    def forward(self, xy, points):
        # points = features
        B, N, C = xy.shape    # [B,N,2]
        S = self.groups       # S
        xy = xy.contiguous()

        # ====Event Sampling====
        # sample_idx = farthest_point_sample(xy, self.groups)      # FPS
        # sample_idx = random_sample(xy, self.groups)              # RS
        _,sample_idx = surface_event_sample(xy, self.groups)       # SES

        new_xy = index_points(xy, sample_idx)                      # [B,S,2]
        new_points = index_points(points, sample_idx)              # [B,S,D]

        # ====Grouping====
        # idx = query_ball(self.radius, self.kneighbors, xy, new_xy)  # Query Ball
        idx = knn_event(self.kneighbors, xy, new_xy)                        # KNN

        grouped_xy = index_points(xy, idx)                                       # [B,S,K,2]
        grouped_points = index_points(points, idx)                               # [B,S,K,D]
        if self.use_xy:
            grouped_points = torch.cat([grouped_points, grouped_xy],dim=-1)      # [B,S,K,2+D]


        if self.normalize is not None:
            #  center on the centroid of grouped events
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)           # [B,S,1,2+D]
            #  center on the sampled events
            if self.normalize =="anchor":
                mean = torch.cat([new_points, new_xy],dim=-1) if self.use_xy else new_points  # [B,S,2+D]
                mean = mean.unsqueeze(dim=-2)                                    # [B,S,1,2+D]
            std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta

        # [B,S,K,2+2*(D)]
        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xy, new_points


#   ==== Event Embedding ====
class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


#   ==== Residual Event Block ====
class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xy=True):
        """
        input: [B,S,K,D]
        output:[B,D,S]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 2+2*channels if use_xy else 2*channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)


    def forward(self, x):
        b, s, k, d = x.size()        # [B,S,K,D]
        x = x.permute(0, 1, 3, 2)    # [B,S,D,K]
        x = x.reshape(-1, d, k)      # [B*S,D,K]
        x = self.transfer(x)         # [B*S,D',K]
        batch_size, _, _ = x.size()
        x = self.operation(x)        # [B*S,D",K]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # [B*S,D"]
        x = x.reshape(b, s, -1).permute(0, 2, 1)              # [B,D",S]
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        return self.operation(x)


class Model(nn.Module):
    def __init__(self, points=2000, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xy=True, normalize="center",
                 dim_expansion=[2, 2], pre_blocks=[2, 2], pos_blocks=[2, 2],
                 radiuses=[0.07, 0.12], k_neighbors=[64, 64], reducers=[2, 2], **kwargs):
        # reducers: down-sampling ratio
        # radiuses: grouping radiuses
        # k_neighbors: grouping neighbors

        super(Model, self).__init__()
        self.stages = len(pre_blocks)
        self.points = points
        self.embedding = ConvBNReLU1D(2, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            radius = radiuses[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, anchor_points, radius, kneighbor, use_xy, normalize)  #[B,S,K,2*(2+D)]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xy=use_xy)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel



    def forward(self, x):
        # x:  [B,N,2]
        xy = x
        x = x.permute(0, 2, 1)          # [B,2,N]
        batch_size, _, _ = x.size()
        x = self.embedding(x)           # [B,D,N]
        for i in range(self.stages):
            xy, x = self.local_grouper_list[i](xy, x.permute(0, 2, 1))  # [B,S,2],[B,S,K,D]
            x = self.pre_blocks_list[i](x)  # [B,D,S]
            x = self.pos_blocks_list[i](x)  # [B,D,S]

        last_feature = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)      # [B,D]

        return last_feature





