## Setup

Clone and enter the project:
```bash
cd ~/Projects/mech-interp-playground
```

Create and activate a virtual environment (only needed once):
```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Every time you sit down to work
```bash
cd ~/Projects/mech-interp-playground
source venv/bin/activate
code .
```

Check the bottom-right corner of VS Code to confirm the interpreter is `./venv/bin/python`.

## Updating requirements.txt
After installing a new package:
```bash
pip freeze > requirements.txt
```

### Advice from someone who knows nothing but is passionately curious 

I'm writing this in a period where all of this could be done in a single prompt flawlessly. 
For anyone learning this, it's easy to fall prey to being over-assisted by LLMs and lose the essence of learning. 
So try to imagine this journey like a PS2 adventure game (that's what I'm doing now), each section is a mini mission that you'll try to accomplish some easy some hard and when you're tired of playing STOP (so that it becomes a fun and memorable activity to return to rather than a task).

## Log

### 2026-08-18 — `gpt2/01_token_embeddings.ipynb`
Implemented token embeddings from scratch.
- Built a toy example with hardcoded values, explored resulting tensor shapes.
- Compared `nn.Embedding` (module-level) vs. raw `torch` tensors — noted `.shape`
  behaves differently between them.
- Implemented `class TokenEmbedding(nn.Module)`.

### 2026-08-18 - `gpt2/02_pos_embedding.ipynb`
- Implemented position embedding from scratch

### 2026-08-20 - `gpt2/03_token__plus_pos.ipynb`
- Implemented the merging of the positonal embeddings and the token embedding.
- Covered the terminology of `overloading` and how it's relevant here.
- Explained with examples of how the `forward()` function is where the bulk of the transformer logic is written out, without making it hard to read through implementation logic.


### 2026-08-21 - `gpt2/04_layer_norm.ipynb`

- Implemented Layer Normalisation
- Expanded on the fundemental data structure `torch.tensor` and how it expands dimensions.

### 2026-08-23 - `gpt2/05_linear_proj.ipynb`

- Implemented the a basic fully connected network to expand the embedding space by `3x` its original size to prepare for Attention mechanism
- Note to self that this name is a bit misleading since it's not the standard projection operation but rather an affine transformation.
  
### 2026-08-23 - `gpt2/06_split_reshape.ipynb`
- Implemented the isolated component which splits the projected data into the Query, Key and Value components.
- Then reshape each of these components to be divided according to the number of attention heads and change the perspective of the dimensions.
- Finally, practiced an implementation from scratch from the begginging of what was learnt.

