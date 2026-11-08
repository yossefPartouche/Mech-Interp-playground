from config import Config
import torch as t
from torch import Tensor
from jaxtyping import Int, Float
import torch.nn as nn
from dataclasses import dataclass
from transformers import GPT2Tokenizer
import einops
import torch.nn.functional as F

class Embed(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.we = nn.Parameter(t.empty(config.d_vocab, config.d_model))
        nn.init.normal_(self.we, std=config.init_range)

    # This is a lookup token is a list of numbers ids and we take parameters embedding 
    # so if token_i = 54 then self.we[54] is a (1, d_model) length embedding
    def forward(self, tokens: Int [Tensor, "batch position"]) -> Float [Tensor, "batch position d_model"]:
        return self.we[tokens]

class PosEmbed(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.pos_W = nn.Parameter(t.empty(config.n_ctx, config.d_model))
        nn.init.normal_(self.pos_W, std=config.init_range)

    # This a local positional embedding where it's purely based on the lengths of the text 
    def forward(self, tokens: Int [Tensor, "batch position"]) -> Float [Tensor, "batch position d_model"]:
        B, P = tokens.shape
        pos = t.arange(P)
        pos_emb = self.pos_W[pos]
        out_unsqueezed = pos_emb.unsqueeze(0)
        out_expanded = out_unsqueezed.expand(B, -1, -1)
        return out_expanded

class LayerNorm(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        self.w = nn.Parameter(t.ones(config.d_model))
        self.b = nn.Parameter(t.zeros(config.d_model))

    def forward(self, residual: Float[Tensor, "batch position d_model"])-> Float[Tensor, "batch position d_model"]:
        mu = residual.mean(dim=-1, keepdim=True)
        std = (residual.var(dim=-1, unbiased=False, keepdim=True) + self.config.layer_norm_eps).sqrt()
        norm = (residual - mu)/std
        norm_parameterised = norm*self.w + self.b

        return norm_parameterised

class Attention(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        # The firt parameter "IGNORE" is the name of the self.register_buffer()
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

    def apply_causal_mask(self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"],
                          )-> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        B, h, tq, tk = attn_scores.shape
        mask = t.tril(t.ones(tq, tk, device=attn_scores.device))
        masked = attn_scores.masked_fill(mask == 0, self.IGNORE)
        return masked

    def forward(self, normalised_resid_pre: Float[Tensor, "batch psn d_model"]) -> Float[Tensor, "batch psn d_model"]:
        q = einops.einsum(normalised_resid_pre, self.W_Q, "batch psn d_model, n_heads d_model d_head -> batch psn n_heads d_head") + self.b_Q
        k = einops.einsum(normalised_resid_pre, self.W_K, "batch psn d_model, n_heads d_model d_head -> batch psn n_heads d_head") + self.b_K
        v = einops.einsum(normalised_resid_pre, self.W_V, "batch psn d_model, n_heads d_model d_head -> batch psn n_heads d_head") + self.b_V
        attn_scores = einops.einsum(q, k, "batch posQ n_heads d_head, batch posK n_heads d_head -> batch n_heads posQ posK")
        attn_scores_scaled = attn_scores / self.config.d_head**0.5
        masked_scores = self.apply_causal_mask(attn_scores_scaled)
        probs = masked_scores.softmax(-1)
        z = einops.einsum(probs, v, "batch n_heads posQ posK, batch posK n_heads d_head -> batch posQ n_heads d_head")
        atn_out = einops.einsum(z, self.W_O, "batch posQ n_heads d_head, n_heads d_head d_model -> batch posQ d_model") + self.b_O
        return atn_out

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
        mlp_in = einops.einsum(normalised_resid_mid, self.W_in, "batch posn d_model, d_model d_mlp -> batch posn d_mlp") + self.b_in
        activation = F.gelu(mlp_in, approximate="tanh")
        mlp_out = einops.einsum(activation, self.W_out, "batch posn d_mlp, d_mlp d_model -> batch posn d_model") + self.b_out
        return mlp_out

class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.ln1 = LayerNorm(config)
        self.attn = Attention(config)
        self.ln2 = LayerNorm(config)
        self.mlp = MLP(config)

    def forward(self, resid_pre: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        res1 = self.attn(self.ln1(resid_pre)) + resid_pre
        res2 = self.mlp(self.ln2(res1)) + res1

        return res2

class Unembed(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.W_U = nn.Parameter(t.empty(config.d_model, config.d_vocab))
        nn.init.normal_(self.W_U, std=config.init_range)
        self.b_U = nn.Parameter(t.zeros(config.d_vocab))

    def forward(self, normalised_resid_final: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_vocab"]:
        return einops.einsum(normalised_resid_final, self.W_U, "batch posn d_model, d_model d_vocab -> batch posn d_vocab") + self.b_U

class DemoTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed = Embed(config)
        self.pos_emb = PosEmbed(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_ln = LayerNorm(config)
        self.unembed = Unembed(config)

    def forward(self, tokens: Int[Tensor, "batch posn"]) -> Float[Tensor, "batch posn d_vocab"]:
        residual = self.embed(tokens) + self.pos_emb(tokens)
        for block in self.blocks:
            residual = block(residual)
        logits = self.unembed(self.final_ln(residual))
        return logits


# --------- As you complete more classes add them above this line ------ 

"""
This is a helper function to keep the work clean
and understand throughout the input and outputs
"""

def show(name, x, n=4):
    print(f"{name:<10} shape={tuple(x.shape)}  mean={x.mean().item():+.3f}  std={x.std().item():.3f}")
    print(f"{'':<10} tok0[:{n}] = {x[0, 0, :n].detach().tolist()}")
    print("="*50)

"""
This checks the and prints out each moving part of the model 
"""
def check_components(tokens, device):
    embed, pos_emb = Embed(config).to(device), PosEmbed(config).to(device)
    ln1, atn1 = LayerNorm(config).to(device), Attention(config).to(device)
    with t.no_grad():
        resid = embed(tokens) + pos_emb(tokens);  show("resid", resid)
        normed = ln1(resid);                      show("ln1", normed)
        attn = atn1(normed);                      show("attn", attn)

"""
This checks a transformer block of the model
"""

def check_block(tokens, device):
    embed, pos_emb = Embed(config).to(device), PosEmbed(config).to(device)
    block = TransformerBlock(config).to(device)
    with t.no_grad():
        resid = embed(tokens) + pos_emb(tokens)
        out = block(resid);  show("block out", out)
        assert out.shape == resid.shape       # a block must preserve shape

"""
This checks the model as a whole
"""

def check_model(tokens, device):
    model = DemoTransformer(config).to(device)
    with t.no_grad():
        logits = model(tokens);  show("logits", logits)


if __name__ == "__main__":
    device = "mps" if t.backends.mps.is_available() else "cpu"
    config = Config()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    text = ["In mathematics, the concept of a measure is a generalization and formalization of geometrical measures",
            "(length, area, volume) and other common notions, such as magnitude, mass, and probability of events.",
            "These seemingly distinct concepts have many similarities and can often be treated together in a single mathematical context.",
            "Measures are foundational in probability theory, integration theory, and can be generalized to assume negative values, as with electrical charge.",
            "Far-reaching generalizations (such as spectral measures and positive operator-valued measures) of measure are widely used in quantum physics and physics in general."]

    tokenizer.pad_token = tokenizer.eos_token
    tokens = (tokenizer(text, return_tensors="pt", padding=True)["input_ids"]).to(device)
    print("pieces:", tokenizer.convert_ids_to_tokens(tokens[0]))
    print("tokens:", tokens.shape)

    print("="*50)
    print("BELOW NOW SHOWS THE SHAPES, VALUES AND BASIC STATS")
    print("="*50)

    check_components(tokens, device)
    check_block(tokens, device)
    check_model(tokens, device)




    