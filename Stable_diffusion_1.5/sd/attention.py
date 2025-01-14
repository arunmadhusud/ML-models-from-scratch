import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):

    def __init__(self,n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        
        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads  
        self.d_head = d_embed // n_heads
    
    def forward(self, x: torch.Tensor, casual_mask=False):
        # x: (Batch, H*W, channels)

        input_shape = x.shape

        batch_size, sequence_length, d_embed = input_shape

        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        q,k,v = self.in_proj(x).chunk(3, dim=-1)
        
        # (Batch, H*W, channels) -> (Batch, H*W, heads, d_head) -> (Batch, heads, H*W, d_head)
        q = q.view(intermim_shape).transpose(1,2)
        k = k.view(intermim_shape).transpose(1,2)
        v = v.view(intermim_shape).transpose(1,2)

        # (Batch, heads, H*W, d_head) -> (Batch, heads, H*W, H*W)
        weight = q @ k.transpose(-1,-2)

        if casual_mask:
            # (H*W, H*W), upper triangular matrix
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)
        
        # (Batch, heads, H*W, H*W) @ (Batch, heads, H*W, d_head) -> (Batch, heads, H*W, d_head)
        output = weight @ v
        
        # (Batch, heads, H*W, d_head) -> (Batch, H*W, heads, d_head)
        output = output.transpose(1,2)

        # (Batch, H*W, heads, d_head) -> (Batch, H*W, channels)
        output = output.reshape(input_shape)

        # (Batch, H*W, channels) -> (Batch, H*W, channels)
        output = self.out_proj(output)

        return output



class CrossAttention(nn.Module):
    def __init__(self,n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)

        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self,x,y):
        # x: (Batch, H*W, channels), This is our Query
        # y: (Batch, sequence_length, d_cross) = (Batch, 77 , 768), This is our Key and Value

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        intermim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Multiply Query with Query Projection
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        # (Batch, H*W, channels) -> (Batch, H*W, heads, d_head) -> (Batch, heads, H*W, d_head)
        q = q.view(intermim_shape).transpose(1,2)
        k = k.view(intermim_shape).transpose(1,2)
        v = v.view(intermim_shape).transpose(1,2)

        weight = q @ k.transpose(-1,-2)
        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1) # no casual mask here

        output = weight @ v

        # (Batch, heads, H*W, d_head) -> (Batch, H*W, heads, d_head)
        output = output.transpose(1,2).contiguous()

        # (Batch, H*W, heads, d_head) -> (Batch, H*W, channels)
        output = output.view(input_shape)

        # (Batch, H*W, channels) -> (Batch, H*W, channels)
        output = self.out_proj(output)

        return output





