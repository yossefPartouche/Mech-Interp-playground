import torch as t
from torch import Tensor
from jaxtyping import Int, Float
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class Config:
    d_model: int = 768 # This is the n_embd
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    n_heads: int = 12
    d_head: int = d_model // n_heads
    d_mlp: int = 4*d_model
    n_layers: int = 12

