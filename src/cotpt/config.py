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
OUTPUT_DIR = "./checkpoints/qwen3-0.6b-cotpt"

# ---- Shared: hidden-thought shape ----
NUM_HIDDEN_THOUGHT_TOKENS = 5   # "thought length" -- used by both inference and training

# ---- Inference (scripts/generate.py, cotpt/inference.py) ----
MAX_VISIBLE_TOKENS = 80
THINKING_TEMPERATURE = 0.8
THINKING_DO_SAMPLE = True
REAL_TOKEN_TEMPERATURE = 0.7
REAL_TOKEN_DO_SAMPLE = False
SHOW_HIDDEN_THOUGHTS = True

# ---- Training (scripts/train.py, cotpt/training.py) ----
NUM_ROLLOUTS = 3             # independent thoughts sampled per position (REINFORCE baseline)
NUM_THINK_POSITIONS = 4      # positions sampled per training example
LOOKAHEAD = 4                # m: real future tokens each thought is scored against
THINK_TEMPERATURE = 1.0      # keep >= 1 during training -- REINFORCE needs exploration

BASE_LR = 5e-6               # small: this is a pretrained model, not a fresh init
MIX_HEAD_LR = 1e-3           # the mixing head starts from scratch, can move faster
VALUE_HEAD_LR = 1e-3
AUX_LM_LOSS_WEIGHT = 1.0     # plain next-token CE, keeps general LM ability from drifting
REINFORCE_LOSS_WEIGHT = 1.0
MIX_LOSS_WEIGHT = 1.0
VALUE_LOSS_WEIGHT = 0.5
MAX_GRAD_NORM = 1.0

NUM_TRAINING_STEPS = 60
LOG_EVERY = 5
GRADIENT_ACCUMULATION_STEPS = 1   # >1 accumulates loss over N examples before optimizer.step()

SEED = 42

# ---- RL stability improvements (all default ON; see README for what each does) ----
NORMALIZE_REWARD = True      # mean, not sum, log-likelihood over the lookahead window
USE_VALUE_BASELINE = True    # per-token credit assignment via a learned critic
USE_KL_PENALTY = True        # penalize the thought policy drifting from a reference distribution
KL_COEFF = 0.05
USE_ENTROPY_BONUS = True     # keep some exploration alive, avoid early collapse to one thought
ENTROPY_COEFF = 0.01

# ---- LoRA (optional; makes the KL reference nearly free via disable_adapter()) ----
USE_LORA = False
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# If USE_KL_PENALTY is on and USE_LORA is off, a full frozen reference copy of
# the model is created instead (roughly doubles model memory). Set False to
# disable the reference model entirely (KL term becomes 0) if that's too
# expensive for your hardware without LoRA.
USE_FROZEN_REFERENCE_WHEN_NO_LORA = True
