## Setup

Needs [uv](https://docs.astral.sh/uv/) installed once on the machine (`brew install uv`).

```bash
cd ~/Projects/Mech-Interp-playground
uv sync
```

This creates `.venv/`, installs the packages listed in `pyproject.toml` (exact versions pinned in
`uv.lock`) and installs the project's own code (`src/mechinterp`). Run it again after pulling changes.

## Project structure

```
src/mechinterp/      reusable, trusted code: models, config, helpers (import as `mechinterp`)
notebooks/           learning: lessons, notes, trying out small functions
  gpt2-pytorch-modules/   GPT-2 step by step with nn.Module
  gpt2-raw-tensors/       GPT-2 step by step with raw tensors
  transformer-lens/       exploring TransformerLens
  scratch/                rough work
experiments/         one folder per question (NNN_short-name/), see experiments/README.md
scripts/             tools that use the model (sanity checks, training)
tools/               workflow helpers (daily_updates.py, not tracked)
DAILY_UPDATES.md     learning log
```

Rules: nothing imports from `notebooks/`; experiments import from `src/` and never from each other;
code moves into `src/` the second time it's needed; modify models with subclasses or config
options instead of copying `model.py`; experiment outputs stay out of git.

## Every time you sit down to work

```bash
cd ~/Projects/Mech-Interp-playground
code .
```

- Interpreter and notebook kernel: `.venv` (no need to activate anything).
- Run a script: `uv run python scripts/playground.py`
- Add a package: `uv add <package>` (updates `pyproject.toml` and `uv.lock`; commit both).

## Learning path

Lessons come from [ARENA](https://github.com/callummcdougall/ARENA_3.0) (kept as read-only reference in
`~/Projects/ARENA_3.0`). Every lesson is implemented here: read it, rebuild it in
`notebooks/arena-<section>-<name>/`, move reused code into `src/`, and turn own questions into experiments.

| ARENA section | Read | Implemented in playground | Experiments |
|---|---|---|---|
| 1.1 Transformer from scratch | ✓ | `gpt2-pytorch-modules/`, `gpt2-raw-tensors/`, `src/mechinterp/models/` | — |
| 1.2 Intro to mech interp | ✓ | `basic_MI_technique/`, `transformer-lens/` | [001](experiments/001_logit-lens-eiffel-tower/) |
| 1.3.1 Linear probes | next | | |
| 1.3.2 Function vectors & steering | | | |
| 1.3.3 SAEs | | | |
| 1.4.1 Indirect object identification | | | |
| 1.5.x (pick one) | | | |

### Advice from someone who knows nothing 

I'm writing this in a period where all of this could be done in a single prompt flawlessly. 
For anyone learning this, it's easy to fall prey to being over-assisted by LLMs and lose the essence of learning. 
So try to imagine this journey like a PS2 adventure game (that's what I'm doing now), each section is a mini mission that you'll try to accomplish some easy some hard and when you're tired of playing STOP (so that it becomes a fun and memorable activity to return to rather than a task).
