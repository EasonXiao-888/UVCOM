import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Optional
from torch import Tensor
from sklearn.cluster import KMeans
from einops import rearrange

def EM_RBF(mu, x,iter):
    '''
    mu [b,k,d]
    x  [b,l,d]
    '''
    em_iter = iter
    # propagation -> make mu as video-specific mu
    norm_x = calculate_l1_norm(x)
    for _ in range(em_iter):
        norm_mu = calculate_l1_norm(mu)
        sigma = 1.2
        latent_z = F.softmax(-0.5 * ((norm_mu[:,:,None,:] - norm_x[:,None,:,:])**2).sum(-1) / sigma**2, dim=1)
        norm_latent_z = latent_z / (latent_z.sum(dim=-1, keepdim=True)+1e-9)
        mu = torch.einsum('nkt,ntd->nkd', [norm_latent_z, x])
    return mu

def calculate_l1_norm(f):
    f_norm = torch.norm(f, p=2, dim=-1, keepdim=True)
    f = f / (f_norm + 1e-9)
    return f

def BMRW(x, y, w):
    x_norm = calculate_l1_norm(x)
    y_norm = calculate_l1_norm(y)
    eye_x = torch.eye(x.size(1)).float().to(x.device)

    latent_z = F.softmax(torch.einsum('nkd,ntd->nkt', [y_norm, x_norm]) * 5.0, 1)
    norm_latent_z = latent_z / (latent_z.sum(dim=-1, keepdim=True) + 1e-9)
    affinity_mat = torch.einsum('nkt,nkd->ntd', [latent_z, norm_latent_z])
    # mat_inv_x, _ = torch.linalg.solve(eye_x, eye_x - (w ** 2) * affinity_mat)
    mat_inv_x, _ = torch.solve(eye_x, eye_x - (w ** 2) * affinity_mat)
    y2x_sum_x = w * torch.einsum('nkt,nkd->ntd', [latent_z, y]) + x
    refined_x = (1 - w) * torch.einsum('ntk,nkd->ntd', [mat_inv_x, y2x_sum_x])    

    return refined_x

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class global_fusion(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0,dim_feedforward=1024,activation="relu"):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src, key,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        src2 = self.multihead_attn(query=self.with_pos_embed(src, pos),
                                #    key=self.with_pos_embed(key, pos),
                                   key = key,
                                   value=key, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        
        src = src * src2
        return src



