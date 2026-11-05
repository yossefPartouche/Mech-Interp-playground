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
    

# --------- As you complete more classes add them above this line ------ 

if __name__ == "__main__":
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

    embed, pos_emb = Embed(config), PosEmbed(config)

    tok_out = embed(tokens)
    pos_out = pos_emb(tokens)
    resid = tok_out + pos_out
    print("tok_out:", tok_out.shape, "| pos_out: ", pos_out.shape, "| resid: ", resid.shape)

    """
        print(len(text))
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokens = tokenizer(text, return_tensors="pt", padding=True)["input_ids"]
        print(tokens)
        print("pieces:", tokenizer.convert_ids_to_tokens(tokens[0]))
        print("tokens:", tokens.shape)

    """



    