"""
All tunable constants live here so inference and training stay in sync and
nothing is duplicated between the two entry points.

Note on MODEL_ID: "Qwen/Qwen3.5-0.8B" is a real, recently-released model, but
it's a hybrid architecture (Gated DeltaNet + Attention layers) whose DeltaNet
layers keep a fixed-size recurrent state rather than a per-token KV cache --
there's no "last N tokens" to slice out of that state the way this project's
eviction logic assumes. MODEL_ID below points at Qwen3-0.6B instead: a small,
dense, ordinary GQA+RoPE transformer where "slice the KV cache along the
sequence dimension" is exactly correct for every layer. Swap this for any
other dense causal LM and the rest of the project works unmodified.
"""

# ---- Model ----
MODEL_ID = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "./checkpoints/qwen3-0.6b-quiet-star"

# ---- Shared: hidden-thought shape ----
NUM_HIDDEN_THOUGHT_TOKENS = 5   # "thought length" -- used by both inference and training

# ---- Inference (scripts/generate.py, quiet_star/inference.py) ----
MAX_VISIBLE_TOKENS = 80
THINKING_TEMPERATURE = 0.8
THINKING_DO_SAMPLE = True
REAL_TOKEN_TEMPERATURE = 0.7
REAL_TOKEN_DO_SAMPLE = False
SHOW_HIDDEN_THOUGHTS = True

# ---- Training (scripts/train.py, quiet_star/training.py) ----
NUM_ROLLOUTS = 3             # independent thoughts sampled per position (REINFORCE baseline)
NUM_THINK_POSITIONS = 4      # positions sampled per training example
LOOKAHEAD = 4                # m: real future tokens each thought is scored against
THINK_TEMPERATURE = 1.0      # keep >= 1 during training -- REINFORCE needs exploration

BASE_LR = 5e-6               # small: this is a pretrained model, not a fresh init
MIX_HEAD_LR = 1e-3           # the mixing head starts from scratch, can move faster
AUX_LM_LOSS_WEIGHT = 1.0     # plain next-token CE, keeps general LM ability from drifting
REINFORCE_LOSS_WEIGHT = 1.0
MIX_LOSS_WEIGHT = 1.0
MAX_GRAD_NORM = 1.0

NUM_TRAINING_STEPS = 60
LOG_EVERY = 5

SEED = 42
