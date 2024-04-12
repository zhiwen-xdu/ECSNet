import numpy as np
import torch
import torch.nn as nn


# Farthest Point Sampling
def farthest_point_sample(xy,nevent):
    """
    输入: xy: AER events, [B, N, 2] nevent: 采样event数
    输出: centroids: 采样到的AER events index, [B,nevent]，这些点是key points,它们作为各自局部区域的centroid
    """
    device = xy.device
    B, N, C = xy.shape      # [32,8000,2]
    centroids = torch.zeros([B,nevent], dtype=torch.long).to(device)     
    distance = torch.ones(B, N).to(device) * 1e10                        
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) 
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(nevent):
        centroids[:, i] = farthest                                    
        centroid = xy[batch_indices, farthest, :].view(B, 1, 2)        
        dist = torch.sum((xy - centroid) ** 2, -1)                     
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]                         
    return centroids
    

# Random Sampling    
def random_sample(xy,nevent):
    """
    输入: xy: events, [B, N, 2] nevent: 采样event数
    输出: sample_idx: 采样到的AER events index, [B,nevent]
    """
    device = xy.device
    B, N, _ = xy.shape
    sample_idx = torch.randint(0, N, (1,nevent), dtype=torch.long).to(device)
    sample_idx = sample_idx.repeat([B, 1])
    return sample_idx
    
    
# Surface Event Sampling
class QuantizationLayer(nn.Module):
    def __init__(self,dim=(128,128)):
        nn.Module.__init__(self)
        self.dim = dim

    def forward(self, xy):
        # input: xy-[B,N,2]
        # return: xy-[B,N,2],vox-[B,H,W]
        device = xy.device
        B,N,C = xy.shape
        W,H = self.dim                                     # [W,H]
        xy = (xy * min(self.dim)).int()                    # [B,N,2]
        x = xy[:,:,0].reshape(-1,)                         # [B*N]
        y = xy[:,:,1].reshape(-1,)                         # [B*N]
        b = torch.arange(B*N, dtype=torch.long).to(device) # [B*N]
        for bi in range(B):
            bi_idx = bi * torch.ones(N, dtype=torch.int32)
            b[bi*N:bi*N+N] = bi_idx

        num_voxels = int(H*W*B)
        vox = xy[0,0,:].new_full([num_voxels,], fill_value=0).to(device)  # [B*H*W]

        idx = (x + W * y + W * H * b).long()
        values = torch.ones(B*N,dtype=torch.int32).to(device)
        vox.put_(idx, values, accumulate=True)
        vox = vox.view(-1, H, W)                                    # [B,H,W]

        return xy,vox



class Surface_Event_Sample(nn.Module):
    def __init__(self,dim=(128,128)):  # dim:W,H
        nn.Module.__init__(self)
        self.dim = dim
        self.quantization_layer = QuantizationLayer(dim=self.dim)

    def sample_batch(self,nevent,xy,vox,batch_sample_idx):
        B,_,_ = xy.shape

        for bi in range(B):
            bi_xy = xy[bi,:,:]          # [N,C]
            bi_vox = vox[bi,:,:]        # [H,W]
            bi_sample_idx = self.sample_single(nevent,bi_xy,bi_vox)  # [nevent,]
            batch_sample_idx[bi,:] = bi_sample_idx            # [B,nevent]

        return batch_sample_idx


    def sample_single(self,nevent,bi_xy,bi_vox):
        device = bi_xy.device
        bi_x = bi_xy[:, 0].long()
        bi_y = bi_xy[:, 1].long()
        idx_global_1 = torch.where(bi_vox[bi_y, bi_x] == 1)[0]
        idx_global_2 = torch.where(bi_vox[bi_y, bi_x] > 1)[0]
        encode_xy = bi_x[idx_global_2] + bi_y[idx_global_2] * 10000
        _, local_idx = np.unique(encode_xy.cpu().numpy(), return_index=True)  
        local_idx = torch.from_numpy(local_idx).to(device)
        idx_global_3 = idx_global_2[local_idx]
        idx_unique = torch.cat([idx_global_1, idx_global_3], dim=0)

        idx_out = idx_unique[torch.randint(0, len(idx_unique), (nevent,), dtype=torch.long).to(device)]

        return idx_out


    def forward(self, xy, nevent):
        # xy -> vox: [B,N,C] -> [B,H,W]
        B,N,C = xy.shape
        device = xy.device
        batch_sample_idx = torch.zeros([B, nevent], dtype=torch.long).to(device)  
        xy,vox = self.quantization_layer.forward(xy)      
        last_sample_idx = self.sample_batch(nevent,xy,vox,batch_sample_idx)      
        xy = torch.true_divide(xy,min(self.dim))
        return xy,last_sample_idx


surface_event_sample = Surface_Event_Sample(dim=(240,180)) # W-H,X-Y
