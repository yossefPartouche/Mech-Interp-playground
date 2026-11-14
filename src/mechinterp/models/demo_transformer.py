from mechinterp.config import Config
import torch as t
from torch import Tensor
from jaxtyping import Int, Float
import torch.nn as nn
import einops
import torch.nn.functional as F


class Embed(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.W_E = nn.Parameter(t.empty(config.d_vocab, config.d_model))
        nn.init.normal_(self.W_E, std=config.init_range)

    def forward(self, tokens: Int[Tensor, "batch posn"]) -> Float[Tensor, "batch posn d_model"]:
        return self.W_E[tokens]


class PosEmbed(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.W_pos = nn.Parameter(t.empty(config.n_ctx, config.d_model))
        nn.init.normal_(self.W_pos, std=config.init_range)

    def forward(self, tokens: Int[Tensor, "batch posn"]) -> Float[Tensor, "batch posn d_model"]:
        B_size, T_length = tokens.shape
        pos = t.arange(T_length, device=tokens.device)
        pos_emb = self.W_pos[pos]
        return pos_emb.unsqueeze(0).expand(B_size, -1, -1)


class LayerNorm(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.w = nn.Parameter(t.ones(config.d_model))
        self.b = nn.Parameter(t.zeros(config.d_model))

    def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        mu = residual.mean(dim=-1, keepdim=True)
        std = (residual.var(dim=-1, unbiased=False, keepdim=True) + self.config.layer_norm_eps).sqrt()
        norm = (residual - mu) / std
        return norm * self.w + self.b


class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.W_Q = nn.Parameter(t.empty(config.n_heads, config.d_model, config.d_head))
        self.W_K = nn.Parameter(t.empty(config.n_heads, config.d_model, config.d_head))
        self.W_V = nn.Parameter(t.empty(config.n_heads, config.d_model, config.d_head))
        self.W_O = nn.Parameter(t.empty(config.n_heads, config.d_head, config.d_model))
        self.b_Q = nn.Parameter(t.zeros(config.n_heads, config.d_head))
        self.b_K = nn.Parameter(t.zeros(config.n_heads, config.d_head))
        self.b_V = nn.Parameter(t.zeros(config.n_heads, config.d_head))
        self.b_O = nn.Parameter(t.zeros(config.d_model))
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32))
        nn.init.normal_(self.W_Q, std=config.init_range)
        nn.init.normal_(self.W_K, std=config.init_range)
        nn.init.normal_(self.W_V, std=config.init_range)
        nn.init.normal_(self.W_O, std=config.init_range)

    def apply_causal_mask(self, attn_scores: Float[Tensor, "batch n_heads posnQ posnK"]) -> Float[Tensor, "batch n_heads posnQ posnK"]:
        b, h, tq, tk = attn_scores.shape
        mask = t.tril(t.ones(tq, tk, device=attn_scores.device))
        return attn_scores.masked_fill(mask == 0, self.IGNORE)

    def forward(self, normalised_resid_pre: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        q = einops.einsum(normalised_resid_pre, self.W_Q, "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head") + self.b_Q
        k = einops.einsum(normalised_resid_pre, self.W_K, "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head") + self.b_K
        v = einops.einsum(normalised_resid_pre, self.W_V, "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head") + self.b_V
        attn_scores = einops.einsum(q, k, "batch posnQ n_heads d_head, batch posnK n_heads d_head -> batch n_heads posnQ posnK")
        attn_scaled = attn_scores / self.config.d_head ** 0.5
        masked_scores = self.apply_causal_mask(attn_scaled)
        probs = masked_scores.softmax(dim=-1)
        z = einops.einsum(probs, v, "batch n_heads posnQ posnK, batch posnK n_heads d_head -> batch posnQ n_heads d_head")
        return einops.einsum(z, self.W_O, "batch posnQ n_heads d_head, n_heads d_head d_model -> batch posnQ d_model") + self.b_O


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.W_in = nn.Parameter(t.empty(config.d_model, config.d_mlp))
        self.W_out = nn.Parameter(t.empty(config.d_mlp, config.d_model))
        self.b_in = nn.Parameter(t.zeros(config.d_mlp))
        self.b_out = nn.Parameter(t.zeros(config.d_model))
        nn.init.normal_(self.W_in, std=config.init_range)
        nn.init.normal_(self.W_out, std=config.init_range)

    def forward(self, normalised_resid_mid: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        in_l = einops.einsum(normalised_resid_mid, self.W_in, "batch posn d_model, d_model d_mlp -> batch posn d_mlp") + self.b_in
        activation = F.gelu(in_l, approximate="tanh")
        return einops.einsum(activation, self.W_out, "batch posn d_mlp, d_mlp d_model -> batch posn d_model") + self.b_out


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.ln1 = LayerNorm(config)
        self.attn = Attention(config)
        self.ln2 = LayerNorm(config)
        self.mlp = MLP(config)

    def forward(self, resid_pre: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
        return self.mlp(self.ln2(resid_mid)) + resid_mid


class Unembed(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.W_U = nn.Parameter(t.empty(config.d_model, config.d_vocab))
        self.b_U = nn.Parameter(t.zeros(config.d_vocab))
        nn.init.normal_(self.W_U, std=config.init_range)

    def forward(self, normalised_resid_final: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_vocab"]:
        return einops.einsum(normalised_resid_final, self.W_U, "batch posn d_model, d_model d_vocab -> batch posn d_vocab") + self.b_U


class DemoTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed = Embed(config)
        self.pos_embed = PosEmbed(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_final = LayerNorm(config)
        self.unembed = Unembed(config)

    def forward(self, tokens: Int[Tensor, "batch posn"]) -> Float[Tensor, "batch posn d_vocab"]:
        x = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            x = block(x)
        return self.unembed(self.ln_final(x))