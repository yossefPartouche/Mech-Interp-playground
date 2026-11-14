import torch as t
from transformers import GPT2Tokenizer
from transformer_lens import HookedTransformer
from mechinterp.models import DemoTransformer
from mechinterp.config import Config
import plotly.graph_objects as go
from torch import Tensor








if __name__ == "__main__":
    device = "mps" if t.backends.mps.is_available() else "cpu"
    config = Config()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    reference_gpt2 = HookedTransformer.from_pretrained(
        "gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
    demo_gpt2 = DemoTransformer(config).to(device)
    demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)

    prompt = "The capital of the country containing the Eiffel Tower is"
    loss = reference_gpt2(prompt, return_type="loss")
    logits: Tensor = reference_gpt2(prompt, return_type="logits")
    prediction = logits.argmax(dim=-1).squeeze()[:-1]
    print("Model Loss: ", loss.item())
    print("prediction shape: ", prediction.shape)
    true_tokens = reference_gpt2.to_tokens(prompt).squeeze()[:1]
    print("True tokens shape", true_tokens.shape)
    is_correct = prediction == true_tokens
    print(demo_gpt2.config)