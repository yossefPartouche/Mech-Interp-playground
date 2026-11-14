import torch as t

@t.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=20, device="cpu", temperature=0.0):
    model.eval()
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    for _ in range(max_new_tokens):
        tokens_cond = tokens[:, -model.config.n_ctx:]
        logits = model(tokens_cond)
        next_logits = logits[0, -1]
        if temperature == 0:
            next_token = next_logits.argmax()
        else:
            probs = (next_logits / temperature).softmax(dim=-1)
            next_token = t.multinomial(probs, num_samples=1)
        tokens = t.cat([tokens, next_token.view(1, 1)], dim=1)
    return tokenizer.decode(tokens[0])