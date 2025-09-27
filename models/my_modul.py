import warnings
from typing import Optional, Tuple

import warnings
from typing import Optional, Tuple

import torch
import math

from .utils import apply_rope_x
import torch
import torch.nn as nn
from torch import Tensor



import math

__all__ = ['MLA']
#-------------------------------------------------
class MLA(nn.Module):
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, d_model, n_heads, max_len=1024, rope_theta=10000.0,device = None,dtype = None) ->None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.d_model = d_model #输入输出维度 embed_dim
        self.n_heads = n_heads #头数 num_heads
        self.dh = d_model // n_heads #每个头的维度 embed_dim//num_heads
        self.q_proj_dim = d_model // 2  #q 的投影
        self.kv_proj_dim = (2 * d_model) // 3 #kv的投影

        self.qk_nope_dim = self.dh // 2 #qk的普通部分
        self.qk_rope_dim = self.dh // 2 #qk参与RoPE旋转编码的部分

        ## Q projections 定义Q的Lora投影
        #W_dq : 将d_model降维到q_proj_dim,也就是q的投影
        #W_uq :将q_proj_dim投影回d_model
        #q_layernorm : layerNorm归一化
        # Lora
        self.W_dq = torch.nn.Parameter(0.01 * torch.randn((d_model, self.q_proj_dim)))
        self.W_uq = torch.nn.Parameter(0.01 * torch.randn((self.q_proj_dim, self.d_model)))
        self.q_layernorm = torch.nn.LayerNorm(self.q_proj_dim)

        ## KV projections 定义KV的Lora投影
        #W_dkv :kv的投影
        #W_ukv ：kv的d_model反投影
        # Lora
        self.W_dkv = torch.nn.Parameter(0.01 * torch.randn((d_model, self.kv_proj_dim + self.qk_rope_dim)))
        self.W_ukv = torch.nn.Parameter(0.01 * torch.randn((self.kv_proj_dim,
                                                            self.d_model + (self.n_heads * self.qk_nope_dim))))
        self.kv_layernorm = torch.nn.LayerNorm(self.kv_proj_dim)

        # output projection 定义输出投影
        self.W_o = torch.nn.Parameter(0.01 * torch.randn((d_model, d_model)))

        # RoPE 旋转位置编码
        self.max_seq_len = max_len
        self.rope_theta = rope_theta

        # https://github.com/lucidrains/rotary-embedding-torch/tree/main
        # visualize emb later to make sure it looks ok
        # we do self.dh here instead of self.qk_rope_dim because its better
        #旋转位置编码 计算过程
        #预计算 cos 和 sin，并存入 register_buffer()
        freqs = 1.0 / (rope_theta ** (torch.arange(0, self.dh, 2).float() / self.dh))
        emb = torch.outer(torch.arange(self.max_seq_len).float(), freqs)
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]

        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
        # This is like a parameter but its a constant so we can use register_buffer
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)



    def forward(self, x, kv_cache=None, past_length=0):
        #batch_size , s= 长度 ，d= dim
        B, S, D = x.size()


        # Q Projections 计算Q
        '''
        先 降维 (W_dq)，再 LayerNorm，然后 恢复原始维度 (W_uq)
        分割成普通部分 (Q) 和 RoPE 部分 (Q_for_rope)
        矩阵乘法
        '''
        compressed_q = x @ self.W_dq
        compressed_q = self.q_layernorm(compressed_q)
        Q = compressed_q @ self.W_uq
        Q = Q.view(B, -1, self.n_heads, self.dh).transpose(1, 2)
        Q, Q_for_rope = torch.split(Q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)

        # Q Decoupled RoPE
        '''
        从 cos_cached, sin_cached 提取 RoPE 旋转信息
        应用 RoPE 变换
        '''
        cos_q = self.cos_cached[:, :, past_length:past_length + S, :self.qk_rope_dim // 2].repeat(1, 1, 1, 2)
        sin_q = self.sin_cached[:, :, past_length:past_length + S, :self.qk_rope_dim // 2].repeat(1, 1, 1, 2)
        Q_for_rope = apply_rope_x(Q_for_rope, cos_q, sin_q)

        # KV Projections
        if kv_cache is None:
            compressed_kv = x @ self.W_dkv
            KV_for_lora, K_for_rope = torch.split(compressed_kv,
                                                  [self.kv_proj_dim, self.qk_rope_dim],
                                                  dim=-1)
            KV_for_lora = self.kv_layernorm(KV_for_lora)
        else:
            new_kv = x @ self.W_dkv
            compressed_kv = torch.cat([kv_cache, new_kv], dim=1)
            new_kv, new_K_for_rope = torch.split(new_kv,
                                                 [self.kv_proj_dim, self.qk_rope_dim],
                                                 dim=-1)
            old_kv, old_K_for_rope = torch.split(kv_cache,
                                                 [self.kv_proj_dim, self.qk_rope_dim],
                                                 dim=-1)
            new_kv = self.kv_layernorm(new_kv)
            old_kv = self.kv_layernorm(old_kv)
            KV_for_lora = torch.cat([old_kv, new_kv], dim=1)
            K_for_rope = torch.cat([old_K_for_rope, new_K_for_rope], dim=1)

        KV = KV_for_lora @ self.W_ukv
        KV = KV.view(B, -1, self.n_heads, self.dh + self.qk_nope_dim).transpose(1, 2)
        K, V = torch.split(KV, [self.qk_nope_dim, self.dh], dim=-1)
        S_full = K.size(2)

        # K Rope
        K_for_rope = K_for_rope.view(B, -1, 1, self.qk_rope_dim).transpose(1, 2)
        cos_k = self.cos_cached[:, :, :S_full, :self.qk_rope_dim // 2].repeat(1, 1, 1, 2)
        sin_k = self.sin_cached[:, :, :S_full, :self.qk_rope_dim // 2].repeat(1, 1, 1, 2)
        K_for_rope = apply_rope_x(K_for_rope, cos_k, sin_k)

        # apply position encoding to each head
        K_for_rope = K_for_rope.repeat(1, self.n_heads, 1, 1)

        # split into multiple heads
        q_heads = torch.cat([Q, Q_for_rope], dim=-1)
        k_heads = torch.cat([K, K_for_rope], dim=-1)
        v_heads = V  # already reshaped before the split

        # make attention mask
        mask = torch.ones((S, S_full), device=x.device)
        mask = torch.tril(mask, diagonal=past_length)
        mask = mask[None, None, :, :]

        sq_mask = mask == 1

        # attention
        x = torch.nn.functional.scaled_dot_product_attention(
            q_heads, k_heads, v_heads,
            attn_mask=sq_mask
        )

        x = x.transpose(1, 2).reshape(B, S, D)

        # apply projection
        x = x @ self.W_o.T

        return x
        #return x, compressed_kv