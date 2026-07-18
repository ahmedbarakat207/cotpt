# quiet-star-prototype

A local PyTorch + Hugging Face `transformers` prototype of **per-token hidden
deliberation with aggressive KV-cache eviction**, plus a **Quiet-STaR-style
think/talk/learn training loop** that gives the model an actual incentive to
make its hidden thoughts useful.

Two things you can do here:

1. **Generate** (`scripts/generate.py`): for every visible output token, the
   model secretly generates a few hidden "thinking" tokens, uses them to pick
   the next real token, then evicts them from the KV cache completely --
   future tokens can never attend to that thinking.
2. **Train** (`scripts/train.py`): fine-tune the model with REINFORCE so its
   hidden thoughts stop being arbitrary continuations and start being tokens
   it has gradient-based incentive to make useful.

## Project structure

```
quiet-star-prototype/
├── pyproject.toml           # pip install -e . installs the package + deps
├── requirements.txt
├── src/quiet_star/
│   ├── config.py            # every tunable constant, one source of truth
│   ├── data.py               # mock prompt + built-in toy training corpus
│   ├── model_utils.py        # device pick, model/tokenizer load, sampling, EOS
│   ├── mixing_head.py         # the "talk head" used in training
│   ├── inference.py           # the eviction generation loop (Steps A-F)
│   └── training.py            # the think/talk/learn training step
├── scripts/
│   ├── generate.py            # CLI: run the eviction loop
│   └── train.py                # CLI: run think/talk/learn training
└── tests/                     # pytest suite, runs against a tiny dummy model
```

## Setup

```bash
pip install -e .
# or: pip install -r requirements.txt
```

Requires `torch` and `transformers>=4.51.0` (needed for Qwen3 support).

## Quickstart

```bash
# Generate with the aggressive-eviction loop
python scripts/generate.py

# Try your own prompt / hyperparameters
python scripts/generate.py --prompt "Q: What is 17 + 26?\nA:" --num-hidden-tokens 8

# Train (writes a checkpoint to ./checkpoints/qwen3-0.6b-quiet-star by default)
python scripts/train.py --num-steps 60

# Generate with the fine-tuned weights
python scripts/generate.py --model-id ./checkpoints/qwen3-0.6b-quiet-star

# Run the test suite (uses a tiny random-weight dummy model, no download needed)
pytest tests/ -v
```

## A note on the model

`Qwen/Qwen3.5-0.8B` is real (released Feb 2026), but it's a **hybrid**
architecture: 24 layers arranged as 6 x (3x Gated-DeltaNet -> 1x Attention).
Only the Attention layers keep a normal per-token KV cache that can be
sliced along the sequence dimension the way this project's eviction logic
does. The DeltaNet layers instead keep a fixed-size *recurrent state* that
has already irreversibly mixed in every past token -- there's no "last N
tokens" to slice out, only a "restore the whole state from before"
operation, which is a different (and more involved) problem.

So `config.MODEL_ID` points at `Qwen/Qwen3-0.6B` instead: a small, dense,
ordinary GQA + RoPE transformer, where "slice the KV cache along dim=-2" is
exactly correct for every layer. Swap `MODEL_ID` for any other dense causal
LM and the rest of the project works unmodified. Wiring up eviction for
Qwen3.5's hybrid cache specifically would be a legitimate, separate project.

Also worth knowing: the `transformers` Cache API has moved past
tuple-of-tuples. It's now `Cache` objects with a `.layers` list, each layer
exposing `.keys`/`.values` tensors and its own `.crop()`. `inference.py` uses
`cache.crop()` (the current, correct, built-in way to do the dim=-2 slice)
and includes `crop_cache_manually()` purely so you can see the raw tensor
slicing under the hood.

## How the eviction loop works (`quiet_star.inference`)

Per visible token:

- **A. Checkpoint** -- remember the cache's current (clean) length.
- **B. Think** -- generate `NUM_HIDDEN_THOUGHT_TOKENS` hidden tokens,
  ordinary autoregressive KV growth, each attending to the ones before it.
- **C. Pick** -- sample/argmax the real next token from the logits produced
  right after the last hidden token.
- **D. Evict** -- `cache.crop(checkpoint_len)`, rolling the cache back to
  exactly its pre-thinking state.
- **E. Commit** -- feed the single real token through the clean cache and
  print it.
- **F.** repeat.

No manual `position_ids`/`cache_position` bookkeeping is needed: `transformers`
derives position from `cache.get_seq_length()`, which reads tensor shape
directly, so once the cache is cropped, positions self-correct automatically.

## How training works (`quiet_star.training`)

Mirrors the Quiet-STaR paper's own framing:

- **Think** -- at a handful of positions per example, sample several
  independent candidate thoughts (rollouts).
- **Talk** -- the `MixingHead` learns how much to blend the post-thought
  hidden state with the plain no-thought hidden state
  (`mixed = (1-w)*before + w*after`), then the blend is projected through
  the model's own `lm_head`.
- **Learn** -- REINFORCE on the thought tokens. Each rollout's reward is how
  much its thought raised the log-likelihood of the next `LOOKAHEAD` real
  tokens, relative to the **mean** reward across that position's rollouts
  (Quiet-STaR's own baseline -- thoughts are judged against their peers).

**Deliberate simplifications vs. the paper**, and why:
- Positions to think at are a random *subset* per example (all rollouts
  *within* one position are still batched together), rather than every
  token in parallel via the paper's custom attention-mask trick. That trick
  is a throughput optimization, not a correctness one -- this trades some
  speed for code you can read top to bottom.
- The built-in corpus (`quiet_star.data.TRAIN_TEXTS`) is 8 short original
  word problems (arithmetic double-checked), not a real dataset. Point it
  at real data for anything beyond confirming the mechanism runs.
- No LoRA/PEFT -- plain full fine-tuning of the (tiny) base model plus the
  mixing head. Fine for 0.6B; wrap `model` with `peft.get_peft_model(...)`
  before building the optimizer if you're memory-constrained.

**Connecting training to inference:** training saves a normal HF checkpoint
plus a separate `mixing_head.pt`. Pointing `generate.py --model-id` at that
checkpoint loads the fine-tuned weights and runs the *same* eviction loop
unmodified -- it never touches the mixing head, since that loop always
fully commits to the post-thought prediction. An inference variant that
actually *uses* the mixing head (to blend or "opt out" of unhelpful
thoughts) would be a genuinely different generation loop and isn't built
here.

## What's actually been validated

No Hugging Face Hub access in the sandbox this was built in, so everything
was checked against a same-architecture tiny random-weight Qwen3 model
instead of the real download, via the `pytest` suite in `tests/`:

- `test_cache_eviction.py` -- the evicted cache is numerically identical
  (float tolerance) to a cache that never saw the hidden tokens at all;
  the public generation function runs end to end, with hidden thoughts
  shown/hidden, and degrades gracefully when `num_hidden_tokens=0`.
- `test_reinforce.py` -- the REINFORCE update, in isolation with synthetic
  rewards, actually concentrates probability on the highest-reward action.
- `test_training_integration.py` -- gradients are finite and populate every
  parameter of both the base model and the mixing head; overfitting one
  fixed example drives total loss *and* the plain next-token loss down
  over 50 steps (proof the rollout -> reward -> REINFORCE -> mixing-loss ->
  optimizer pieces are wired together correctly, not just individually
  plausible); short sequences with no valid think positions fall back to
  the plain LM loss instead of crashing.

**What this does *not* prove:** that a real training run, on real data, for
many more steps, produces a model that reasons better. That's a genuinely
larger claim needing real data and evaluation against a no-thinking
baseline to back up.
