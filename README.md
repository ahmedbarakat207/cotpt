# cotpt

A local PyTorch + Hugging Face `transformers` prototype of **per-token hidden
deliberation with aggressive KV-cache eviction**, a **COTPT-style
think/talk/learn training loop** with several RL stability improvements, an
**evaluation harness** to check whether any of it actually helps, and the
engineering around all three (LoRA, dataset loading, logging, checkpoint
resume).

Three things you can do here:

1. **Generate** (`scripts/generate.py`): for every visible output token, the
   model secretly generates a few hidden "thinking" tokens, uses them to pick
   the next real token, then evicts them from the KV cache completely.
2. **Train** (`scripts/train.py`): fine-tune the model with REINFORCE so its
   hidden thoughts stop being arbitrary continuations and start being tokens
   it has gradient-based incentive to make useful.
3. **Evaluate** (`scripts/evaluate.py`): compare no-thinking, evicted hidden
   deliberation, and plain visible chain-of-thought on the same metric, so
   "does this help" has an actual answer instead of an assumption.

## Project structure

```
cotpt/
├── pyproject.toml, requirements.txt
├── src/cotpt/
│   ├── config.py            # every tunable constant, one source of truth
│   ├── data.py                # mock prompt + built-in toy training corpus
│   ├── eval_data.py            # held-out eval problems (disjoint from data.py)
│   ├── dataset.py               # local .txt / HF `datasets` loading for real data
│   ├── model_utils.py            # device pick, model/tokenizer load, sampling, EOS
│   ├── mixing_head.py             # the "talk head" -- blends with/without-thought predictions
│   ├── value_head.py               # per-token REINFORCE baseline (actor-critic)
│   ├── inference.py                 # eviction loop (Steps A-F) + adaptive/mixing-head variant
│   ├── training.py                   # think/talk/learn, with value baseline / KL / entropy
│   ├── evaluation.py                  # no_think vs hidden_deliberation vs visible_cot
│   ├── checkpoint_utils.py             # save/resume: model (or LoRA adapter), heads, optimizer, step
│   └── logging_utils.py                 # JSONL (+ optional wandb) metrics logging
├── scripts/
│   ├── generate.py           # CLI: eviction loop, optionally with the trained mixing head
│   ├── train.py                # CLI: think/talk/learn training, all features flag-controlled
│   └── evaluate.py              # CLI: the 3-condition comparison
└── tests/                     # pytest suite, runs against a tiny dummy model, no download needed
```

## Setup

```bash
pip install -e .                 # core (torch, transformers)
pip install -e ".[lora]"         # + peft/accelerate, for --use-lora
pip install -e ".[data]"         # + datasets, for --hf-dataset
pip install -e ".[tracking]"     # + wandb, for --use-wandb
pip install -e ".[all]"          # everything, including pytest
```

## Quickstart

```bash
# Generate with the aggressive-eviction loop
python scripts/generate.py

# Train (bare defaults: value baseline + KL penalty (frozen reference copy) +
# entropy bonus all on, no LoRA -- see "What each flag does" below)
python scripts/train.py --num-steps 60

# Train cheaper: LoRA + the free disable_adapter() KL reference
python scripts/train.py --use-lora --num-steps 200

# Generate with the fine-tuned weights, using the trained mixing head and
# skipping thinking when the model's already confident
python scripts/generate.py --model-id ./checkpoints/qwen3-0.6b-cotpt \
    --use-mixing-head --entropy-threshold 2.0

# Does any of this help? Compare against the two baselines it needs to beat
python scripts/evaluate.py --model-id ./checkpoints/qwen3-0.6b-cotpt --verbose

# Real data instead of the 8 built-in toy problems
python scripts/train.py --data-path ./my_corpus.txt --num-steps 500

# Resume a run
python scripts/train.py --resume-from ./checkpoints/qwen3-0.6b-cotpt --num-steps 500

# Run the test suite (tiny random-weight dummy model, no download needed)
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
operation, a different (and more involved) problem.

So `config.MODEL_ID` points at `Qwen/Qwen3-0.6B` instead: a small, dense,
ordinary GQA + RoPE transformer, where "slice the KV cache along dim=-2" is
exactly correct for every layer. Swap `MODEL_ID` for any other dense causal
LM and the rest of the project works unmodified.

Also worth knowing: the `transformers` Cache API has moved past
tuple-of-tuples. It's now `Cache` objects with a `.layers` list, each layer
exposing `.keys`/`.values` tensors and its own `.crop()`.

## How the eviction loop works (`cotpt.inference`)

Per visible token: **A. Checkpoint** the cache's current length. **B. Think**
-- generate `NUM_HIDDEN_THOUGHT_TOKENS` hidden tokens, ordinary autoregressive
KV growth. **C. Pick** -- sample/argmax the real next token from the logits
right after the last hidden token. **D. Evict** -- `cache.crop(checkpoint_len)`.
**E. Commit** -- feed the real token through the clean cache and print it.
**F.** repeat.

No manual `position_ids` bookkeeping is needed: position is derived from
`cache.get_seq_length()`, which reads tensor shape directly, so cropping
alone makes positions self-correct.

`generate_with_adaptive_deliberation` is a second variant that actually uses
the trained `MixingHead` (`generate_with_hidden_deliberation` never touches
it -- it always fully commits to the post-thought prediction). It blends the
"no thought" and "post thought" hidden states with the head's own learned
weight `w`, and, if `--entropy-threshold` is set, can skip thinking entirely
when the base prediction is already confident -- a real compute saving, not
just a quality change.

## How training works (`cotpt.training`)

Mirrors the paper's own framing -- **Think**: sample several independent
candidate thoughts (rollouts) at a handful of positions per example. **Talk**:
the `MixingHead` blends the post-thought hidden state with the no-thought one,
projected through the model's own `lm_head`. **Learn**: REINFORCE on the
thought tokens.

**RL stability additions on top of the bare-bones version** (all flag-controlled,
defaults in `config.py`):

| Feature | Flag | What it does |
|---|---|---|
| Reward normalization | always on | mean, not sum, log-likelihood over the lookahead window, so reward scale doesn't depend on `LOOKAHEAD` |
| Value baseline | `--use-value-baseline` (default on) | a learned `ValueHead` gives each thought **token** its own baseline instead of every token in a rollout sharing one scalar mean-of-rollouts baseline -- lower-variance, more precise credit assignment |
| KL penalty | `--use-kl-penalty` (default on) | penalizes the thought policy for drifting from a reference distribution, discouraging reward-hacking into degenerate token sequences. Reference is either a frozen deep-copy of the model, or -- far cheaper -- a LoRA model's own base weights via `model.disable_adapter()` when `--use-lora` is also set |
| Entropy bonus | `--use-entropy-bonus` (default on) | keeps some exploration alive so REINFORCE doesn't collapse onto one low-diversity thought early |

**A real gotcha found while validating this**: once the value baseline is on,
watching raw `total_loss` is misleading. `advantage = reward - value_head(state)`
shrinks toward 0 as the critic gets calibrated (the point -- lower-variance
gradients), which makes the REINFORCE term's magnitude move *toward* zero
from wherever it started, which can make `total_loss` go up even though
training is healthy. Watch `aux_lm_loss` (down), `value_loss` (down), and
`mean_reward` (up) instead -- that's what `scripts/train.py`'s logging and
`tests/test_rl_improvements.py` actually check.

**Deliberate simplifications vs. the paper**, and why:
- Positions to think at are a random *subset* per example (all rollouts
  *within* one position are still batched together), rather than every token
  in parallel via the paper's custom attention-mask trick -- a throughput
  optimization, not a correctness one.
- The built-in corpus (`cotpt.data.TRAIN_TEXTS`) is 8 short original word
  problems (arithmetic double-checked), not a real dataset. Use `--data-path`
  or `--hf-dataset` for real data.

**Engineering**: `--use-lora` wraps the model with `peft` (also makes the KL
reference nearly free); `--gradient-accumulation-steps` for a larger
effective batch without more memory; `--checkpoint-every N` for periodic
saves, `--resume-from <dir>` to continue a run (restores model/adapter
weights, both heads, optimizer state, and step count); `--data-path`/
`--hf-dataset` for real data instead of the toy corpus; `--use-wandb` mirrors
metrics to Weights & Biases if installed, and JSONL logging
(`<output-dir>/train_log.jsonl`) always happens regardless.

## Evaluation (`cotpt.evaluation`)

The one question none of the mechanism-correctness testing above answers:
does any of this actually help? `run_evaluation` compares mean per-token
log-likelihood of a known-correct worked solution under three conditions on
the same held-out problems (`cotpt.eval_data`, disjoint from the
training corpus):

- `no_think` -- plain teacher-forced likelihood, no thinking.
- `hidden_deliberation` -- the same checkpoint/think/evict/commit mechanism
  as generation, but scoring the real ground-truth token instead of sampling.
- `visible_cot` -- no hidden mechanism at all, just a "let's think step by
  step" suffix appended to the visible prompt. **This is the baseline hidden
  deliberation actually needs to beat** -- if plain prompting does just as
  well, the eviction machinery isn't earning its complexity or compute.

Run `scripts/evaluate.py` before and after training on the same eval set to
see whether training moved anything, and don't over-read a result from an
untrained model, a lightly-trained run, or this small a held-out set (6
problems) -- it's a harness, not a benchmark.

## What's actually been validated

No Hugging Face Hub access in the sandbox this was built in, so everything
below was checked against a same-architecture tiny random-weight Qwen3 model
via the `pytest` suite in `tests/`, not the real download:

- **Eviction is bit-exact**: the evicted cache is numerically identical
  (float tolerance) to a cache that never saw the hidden tokens at all, with
  and without the mixing head in the loop (`test_cache_eviction.py`,
  `test_adaptive_inference.py`).
- **REINFORCE direction is correct**: in isolation with synthetic rewards, it
  concentrates probability on the highest-reward action; the value-baseline
  version does the same while also learning to predict rewards accurately
  (`test_reinforce.py`, `test_rl_improvements.py`).
- **KL penalty is exactly 0** when the policy equals the reference (true at
  LoRA init, since LoRA's B matrix starts at zero, and true for a freshly
  deep-copied frozen reference) -- both mechanisms checked
  (`test_rl_improvements.py`).
- **Entropy bonus gradient is correct in isolation**: with every other loss
  term removed, maximizing entropy alone drives it toward the theoretical
  ceiling `log(vocab_size)` (`test_rl_improvements.py`) -- a cleaner,
  non-flaky test than trying to see the effect through a full noisy training
  run, where the other three loss terms competing for the same parameters
  swamp the (intentionally small) entropy signal within a short run.
- **Full pipeline convergence**: overfitting one fixed example drives
  `aux_lm_loss`, `value_loss` down and `mean_reward` up over 50 steps with
  every feature on simultaneously (value baseline + KL + entropy bonus)
  (`test_rl_improvements.py`).
- **LoRA integration**: cache/hidden_states/lm_head/gradients behave
  identically through a `peft`-wrapped model as a plain one; only the LoRA
  parameters receive gradients (`test_full_step_with_lora`).
- **Checkpoint round-trip**: reloaded model weights (plain and LoRA adapter),
  mixing/value heads, and optimizer state all match exactly; resumed LoRA
  models produce bit-identical logits to the original (`test_checkpoint_resume.py`).
- **Evaluation harness self-consistency**: with `num_hidden_tokens=0`, the
  hidden-deliberation eval metric is required to exactly equal the plain
  no-think metric (mechanically, thinking zero steps must degenerate to no
  thinking at all) -- a correctness check independent of whether the
  paradigm itself helps (`test_evaluation.py`).
- **Dataset/logging utilities**: local-file loading, paragraph splitting,
  minimum-length filtering, and JSONL round-trips all tested directly
  (`test_dataset_and_logging.py`). The `--hf-dataset` path is written
  defensively (guarded import, clear error message) but wasn't run against a
  real download -- no Hub access in this sandbox.

**What none of this proves**: that a real training run, on real data, for
many more steps, produces a model that reasons better than the no-thinking
or visible-CoT baselines. That's what `scripts/evaluate.py` exists to check
on your own run -- this project validates the mechanism, not the outcome.
