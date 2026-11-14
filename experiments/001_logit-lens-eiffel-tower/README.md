# 001 — Logit lens on "The Eiffel Tower is in the country whose capital is"

**Question:** At which layer does GPT-2 small start predicting " Paris" / " France"?

**Setup:** `DemoTransformer` loaded with GPT-2 small weights from TransformerLens. Each layer's
residual stream is decoded through `ln_final` + `unembed` (logit lens). See `run.py` for the
plotting helpers (top-k per layer, heatmap, table, token ranks).

**Run:** `python experiments/001_logit-lens-eiffel-tower/run.py`

**Result:** _to fill in_

**Commit it ran on:** _to fill in (`git rev-parse --short HEAD`)_
