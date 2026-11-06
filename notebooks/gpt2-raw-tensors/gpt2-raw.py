from config import Config
import torch as t
from torch import Tensor
from jaxtyping import Int, Float
import torch.nn as nn
from dataclasses import dataclass
from transformers import GPT2Tokenizer

cfg = Config()

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
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device))
    def apply_causal_mask(self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"],
                          )-> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        B, h, tq, tk = attn_scores.shape
        mask = t.tril(t.ones(tq, tk, device=attn_scores.device))
        masked = attn_scores.masked_fill(mask == 0, self.IGNORE)
        return masked

    

# --------- As you complete more classes add them above this line ------ 

"""
This is a helper function to keep the work clean
and understand throughout the input and outputs
"""

def show(name, x, n=4):
    print(f"{name:<10} shape={tuple(x.shape)}  mean={x.mean().item():+.3f}  std={x.std().item():.3f}")
    print(f"{'':<10} tok0[:{n}] = {x[0, 0, :n].detach().tolist()}")
    print("="*50)


if __name__ == "__main__":
    device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
    config = Config()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    text = ["In mathematics, the concept of a measure is a generalization and formalization of geometrical measures",
            "(length, area, volume) and other common notions, such as magnitude, mass, and probability of events.",
            "These seemingly distinct concepts have many similarities and can often be treated together in a single mathematical context.",
            "Measures are foundational in probability theory, integration theory, and can be generalized to assume negative values, as with electrical charge.",
            "Far-reaching generalizations (such as spectral measures and positive operator-valued measures) of measure are widely used in quantum physics and physics in general."]

    tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer(text, return_tensors="pt", padding=True)["input_ids"]
    print("pieces:", tokenizer.convert_ids_to_tokens(tokens[0]))
    print("tokens:", tokens.shape)

    print("="*50)
    print("BELOW NOW SHOWS THE SHAPES, VALUES AND BASIC STATS")
    print("="*50)

    """
    Here you initialise the modules as you complete them
    """
    embed = Embed(config)
    pos_emb = PosEmbed(config)

    with t.no_grad():
        tok_out = embed(tokens);    show("Token Embed", tok_out)
        pos_out = pos_emb(tokens);   show("Position Embed", pos_out)
        resid = tok_out + pos_out;  show("residual", resid)



    