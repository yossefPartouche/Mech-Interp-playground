import torch as t
from transformers import GPT2Tokenizer
from transformer_lens import HookedTransformer
from mechinterp.models import DemoTransformer
from mechinterp.config import Config
import plotly.graph_objects as go

@t.no_grad()
def plot_top_tokens(model, tokenizer, prompt, device, k=5, metric="prob", extra_words=(" France", " Paris")):
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    x = model.embed(tokens) + model.pos_embed(tokens)
    stages = [x]
    for block in model.blocks:
        x = block(x)
        stages.append(x)

    all_logits = t.stack([model.unembed(model.ln_final(s))[0, -1] for s in stages])
    all_probs = all_logits.softmax(-1)

    top_ids = set(all_logits.topk(k, dim=-1).indices.flatten().tolist())
    top_ids |= {tokenizer.encode(w)[0] for w in extra_words}   # always include these

    layers = list(range(len(stages)))
    fig = go.Figure()
    for tid in sorted(top_ids):
        label = repr(tokenizer.decode([tid]))
        if metric == "prob":
            y = all_probs[:, tid].cpu().tolist()
        else:
            y = (all_logits > all_logits[:, tid].unsqueeze(1)).sum(-1).cpu().tolist()
        fig.add_trace(go.Scatter(x=layers, y=y, mode="lines+markers", name=label))

    fig.update_layout(
        title=prompt,
        xaxis_title="layer (0 = embeddings)",
        yaxis_title="probability" if metric == "prob" else "rank (log, lower = better)",
        hovermode="x unified",
        width=1000, height=600,
    )
    if metric == "rank":
        fig.update_yaxes(type="log", autorange="reversed")
    fig.show()

@t.no_grad()
def top_tokens_heatmap(model, tokenizer, prompt, device, k=40, start_layer=6):
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    x = model.embed(tokens) + model.pos_embed(tokens)
    stages = [x]
    for block in model.blocks:
        x = block(x)
        stages.append(x)

    layers = list(range(start_layer, len(stages)))
    probs = t.stack([model.unembed(model.ln_final(stages[l]))[0, -1].softmax(-1) for l in layers])  # (layers, vocab)

    # tokens that are in the top-k at any chosen layer
    ids = sorted(set(probs.topk(k, dim=-1).indices.flatten().tolist()),
                 key=lambda i: -probs[-1, i].item())        # order by final-layer probability
    z = probs[:, ids].T.cpu().numpy()                       # (tokens, layers)
    labels = [repr(tokenizer.decode([i])) for i in ids]

    fig = go.Figure(go.Heatmap(z=z, x=layers, y=labels, colorscale="Viridis",
                               colorbar=dict(title="prob")))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=prompt, xaxis_title="layer", height=max(600, 18 * len(ids)))
    fig.show()

@t.no_grad()
def top_tokens_table(model, tokenizer, prompt, device, k=40, start_layer=6):
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    x = model.embed(tokens) + model.pos_embed(tokens)
    stages = [x]
    for block in model.blocks:
        x = block(x)
        stages.append(x)

    for layer in range(start_layer, len(stages)):
        logits = model.unembed(model.ln_final(stages[layer]))[0, -1]
        top = logits.topk(k).indices.tolist()
        print(f"\nlayer {layer}:")
        print([tokenizer.decode([i]) for i in top])

@t.no_grad()
def next_token_preds(model, tokenizer, prompt, k=10):
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    logits = model(tokens)[0, -1]
    probs = logits.softmax(-1)
    top = probs.topk(k)
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"{p:6.2%}  {tokenizer.decode([i])!r}")

@t.no_grad()
def all_token_ranks(model, tokenizer, prompt, device):
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    x = model.embed(tokens) + model.pos_embed(tokens)
    stages = [x]
    for block in model.blocks:
        x = block(x)
        stages.append(x)

    logits = t.stack([model.unembed(model.ln_final(s))[0, -1] for s in stages])  # (13, vocab)
    # rank of each token within its layer: 0 = top
    order = logits.argsort(dim=-1, descending=True)
    ranks = t.empty_like(order)
    ranks.scatter_(1, order, t.arange(logits.size(1), device=logits.device).expand_as(order))
    print(ranks.shape)
    return ranks.cpu()          # ranks[layer, token_id]

if __name__ == "__main__":
    device = "mps" if t.backends.mps.is_available() else "cpu"
    config = Config()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    reference_gpt2 = HookedTransformer.from_pretrained(
        "gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
    demo_gpt2 = DemoTransformer(config).to(device)
    demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)
    print(demo_gpt2.config)

    # call it here, after the variables exist
    """"
    rank_by_layer(demo_gpt2, tokenizer,
                  "The Eiffel Tower is in the country whose capital is",
                  [" France", " Paris"], device)
    """
    #ranks = all_token_ranks(demo_gpt2, tokenizer, "The Eiffel Tower is in the country whose capital is", device)
    #plot_top_tokens(demo_gpt2, tokenizer, "The Eiffel Tower is in the country whose capital is", device, k=40)
    #top_tokens_heatmap(demo_gpt2, tokenizer, "The Eiffel Tower is in the country whose capital is", device=device)
    #top_tokens_table(demo_gpt2, tokenizer, "The Eiffel Tower is in the country whose capital is", device=device, k=10)
    #next_token_preds(demo_gpt2, tokenizer, "The Eiffer Tower is the country whose captial is")