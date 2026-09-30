"""
Configuration for fine-tuning `ettin-encoder-17m` on banking77.

Every parameter used by train.py and predict.py lives here.
Relative paths are resolved from this file's folder, so the scripts can be
launched from any working directory.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
DATA_DIR = PROJECT_DIR / "data"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
CATEGORIES_FILE = DATA_DIR / "categories.json"

# Pretrained checkpoint (PyTorch weights: pytorch_model.bin + tokenizer files).
# The onnx/ subfolder is not used for training.
MODEL_DIR = PROJECT_DIR / "ettin-encoder-17m"

# Every run of train.py gets its own folder inside OUTPUTS_DIR, named
# <RUN_NAME_PREFIX>_<date>_<time>, e.g. outputs/run_2026-09-30_14-05-12/
OUTPUTS_DIR = PROJECT_DIR / "outputs"
RUN_NAME_PREFIX = "run"
RUN_TIMESTAMP_FORMAT = "%Y-%m-%d-%H%M%S"   # local time; sorts chronologically

# File names inside each run folder.
BEST_MODEL_DIRNAME = "best_model"                         # model + tokenizer
HISTORY_FILENAME = "history.json"                         # per-epoch metrics
METRICS_FILENAME = "metrics.json"                         # final test metrics
TEST_REPORT_FILENAME = "test_classification_report.txt"   # per-class test scores
PLOT_FILENAME = "training_curves.png"                     # loss and metric graphs
CONFIG_COPY_FILENAME = "config.py"                        # exact copy of this file

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
# Column names in train.csv / test.csv (confirmed: header is "text,category",
# labels are category names such as "card_arrival"). If a name is not found,
# the first matching name from the candidate lists is used instead.
# categories.json is a list of 77 names; its order defines the label ids.
TEXT_COLUMN = "text"
LABEL_COLUMN = "category"
TEXT_COLUMN_CANDIDATES = ("text", "sentence", "utterance", "query")
LABEL_COLUMN_CANDIDATES = ("category", "label", "intent", "labels")

# Fraction of train.csv held out (stratified) for model selection and early
# stopping. test.csv is only used once, at the very end.
# Set to 0 to train on all of train.csv and keep the last epoch.
VALIDATION_SPLIT = 0.1

# Maximum tokens per query. banking77 queries are short, and batches are padded
# only to their longest query, so a generous limit costs nothing.
# train.py reports how many queries were truncated; raise this if it isn't 0.
MAX_LENGTH = 128

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
ATTN_IMPLEMENTATION = "sdpa"   # "sdpa" or "eager" (flash-attention is GPU-only)
# The checkpoint's config.json already uses "mean" pooling; set here explicitly
# so the choice is visible. "cls" is the other option. None = checkpoint value.
CLASSIFIER_POOLING = "mean"
# The checkpoint's config.json uses 0.0. 0.1 adds light regularisation to the
# new classification head. None = checkpoint value.
CLASSIFIER_DROPOUT = 0.1

# --------------------------------------------------------------------------- #
# Optimisation
# --------------------------------------------------------------------------- #
NUM_EPOCHS = 15
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 128
LEARNING_RATE = 8e-5
WEIGHT_DECAY = 0.01
ADAM_BETAS = (0.9, 0.98)
ADAM_EPSILON = 1e-6
WARMUP_RATIO = 0.1             # fraction of total steps used for LR warm-up
LR_SCHEDULER = "linear"        # "linear" or "cosine"
MAX_GRAD_NORM = 1.0            # None disables gradient clipping
LABEL_SMOOTHING = 0

# --------------------------------------------------------------------------- #
# Model selection / early stopping
# --------------------------------------------------------------------------- #
METRIC_FOR_BEST_MODEL = "accuracy"  # "accuracy" or "macro_f1" (on validation)
EARLY_STOPPING_PATIENCE = None         # epochs without improvement; None disables

# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
SEED = 8137
DEVICE = "cpu"
NUM_THREADS = None             # None = PyTorch default (uses all cores)
NUM_WORKERS = 0                # DataLoader workers
LOG_EVERY_N_STEPS = 50

# --------------------------------------------------------------------------- #
# Training-curves plot (training_curves.png)
# --------------------------------------------------------------------------- #
# Each graph's plotting area is drawn at 4:3 (width:height). This sets the
# width of one graph in inches; its height is 3/4 of it.
PLOT_GRAPH_WIDTH_INCHES = 6.4
PLOT_DPI = 150

# --------------------------------------------------------------------------- #
# Prediction (predict.py)
# --------------------------------------------------------------------------- #
# predict.py uses the newest run in OUTPUTS_DIR unless --run is given.
PREDICT_TOP_K = 5
