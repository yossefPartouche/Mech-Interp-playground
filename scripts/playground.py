import torch as t
from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer

from mechinterp.config import Config
from mechinterp.models.demo_transformer import (Embed, PosEmbed, LayerNorm, Attention,
                   MLP, TransformerBlock, Unembed, DemoTransformer)
from mechinterp.generate import generate


def show(name, x, n=4):
    print(f"{name:<10} shape={tuple(x.shape)}  mean={x.mean().item():+.3f}  std={x.std().item():.3f}")
    print(f"{'':<10} tok0[:{n}] = {x[0, 0, :n].detach().tolist()}")
    print("=" * 50)


def check_components(config, tokens, device):
    embed, pos_emb = Embed(config).to(device), PosEmbed(config).to(device)
    ln1, atn1 = LayerNorm(config).to(device), Attention(config).to(device)
    with t.no_grad():
        resid = embed(tokens) + pos_emb(tokens);  show("resid", resid)
        normed = ln1(resid);                      show("ln1", normed)
        attn = atn1(normed);                      show("attn", attn)


def check_block(config, tokens, device):
    embed, pos_emb = Embed(config).to(device), PosEmbed(config).to(device)
    block = TransformerBlock(config).to(device)
    with t.no_grad():
        resid = embed(tokens) + pos_emb(tokens)
        out = block(resid);  show("block out", out)
        assert out.shape == resid.shape


def check_model(config, tokens, device):
    model = DemoTransformer(config).to(device)
    with t.no_grad():
        logits = model(tokens);  show("logits", logits)


if __name__ == "__main__":
    device = "mps" if t.backends.mps.is_available() else "cpu"
    config = Config()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # load pretrained weights into your model
    reference_gpt2 = HookedTransformer.from_pretrained(
        "gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
    demo_gpt2 = DemoTransformer(config).to(device)
    result = demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)
    print("missing keys:", result.missing_keys)

    # generation
    print(generate(demo_gpt2, tokenizer, "In mathematics, the concept of", 20, device))

    # per-section checks on a batch
    text = ["In mathematics, the concept of a measure is a generalization and formalization of geometrical measures",
            "(length, area, volume) and other common notions, such as magnitude, mass, and probability of events."]
    tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer(text, return_tensors="pt", padding=True)["input_ids"].to(device)

    check_components(config, tokens, device)
    check_block(config, tokens, device)
    check_model(config, tokens, device)