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

### Log

## Log

### 2026-08-18 — `gpt2/01_token_embeddings.ipynb`
Implemented token embeddings from scratch.
- Built a toy example with hardcoded values, explored resulting tensor shapes.
- Compared `nn.Embedding` (module-level) vs. raw `torch` tensors — noted `.shape`
  behaves differently between them.
- Implemented `class TokenEmbedding(nn.Module)`.

### 2026-08-18 - `gpt2/02_pos_embedding.ipynb`
Implemented position embedding from scratch
