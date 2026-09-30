
# Fine-tuning ettin-encoder-17m on banking77 (CPU)

Fine-tunes the 17M-parameter [Ettin encoder](https://huggingface.co/jhu-clsp/ettin-encoder-17m)
(a ModernBERT-architecture model) to classify customer banking queries into the
77 intents of the banking77 dataset. Everything runs on CPU with PyTorch; no
NVIDIA/CUDA packages are installed.

## Project layout

```
.
├── README.md
├── requirements.txt        # dependencies (CPU-only torch)
├── config.py               # every training parameter and file path
├── train.py                # fine-tuning script
├── predict.py              # 10 example sentences for a manual check
├── outputs/                # created by train.py: one folder per run
│   └── run_<date>_<time>/  # see "Outputs" below
├── data/
│   ├── categories.json     # list of the 77 category names (order = label ids)
│   ├── train.csv           # columns: text,category
│   └── test.csv            # columns: text,category
└── ettin-encoder-17m/      # pretrained checkpoint
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json, tokenizer_config.json, special_tokens_map.json
    └── onnx/               # not used for training
```

## Requirements

- Linux (tested target: Ubuntu, x86-64)
- Python 3.12 (transformers and torch both require Python 3.10 or newer)
- No GPU needed

Pinned versions:

| Package | Version | Why |
| --- | --- | --- |
| torch | `2.11.0+cpu` | CPU-only build from the PyTorch CPU index. 2.10 or newer is needed for the `torch.load` fix CVE-2026-24747, which matters because the weights are a `.bin` file. |
| transformers | `>=5.17,<6` | ModernBERT support; the APIs used were checked against the v5 source. |
| numpy, pandas, scikit-learn | recent | Data loading, splitting and metrics. |
| matplotlib | `>=3.8` | The training-curves PNG. |

## Installation

Ubuntu blocks `pip install` outside a virtual environment, so create one first:

```bash
sudo apt install python3.12-venv      # only if "python3 -m venv" fails
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -U "huggingface_hub"
hf download jhu-clsp/ettin-encoder-17m --local-dir ettin-encoder-17m/
mkdir data
cd data
wget https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/refs/heads/master/banking_data/categories.json
wget https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/refs/heads/master/banking_data/test.csv
wget https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/refs/heads/master/banking_data/train.csv
```

Check that the CPU build was installed (expect `2.11.0+cpu None`):

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

## Training

```bash
python train.py
```

What it does:

1. Creates a new run folder, `outputs/run_<date>_<time>/`, and copies
   `config.py` into it before anything else, so even an interrupted run keeps
   its settings.
2. Reads `categories.json` and both CSV files, and maps category names to label ids.
3. Holds out 10% of `train.csv` (stratified by category) as a validation set.
   `test.csv` is only used once, at the end.
4. Prints checks: whether all 77 categories appear in each file, the longest
   query in tokens, and how many queries are truncated at `MAX_LENGTH`.
5. Loads the checkpoint with a new classification head (mean pooling).
6. Trains with AdamW, linear warm-up and decay, and gradient clipping.
7. After each epoch: evaluates on the validation set, saves the model if it is
   the best so far, updates `history.json` and redraws `training_curves.png`.
   Stops early after 3 epochs without improvement.
8. Reloads the best model, evaluates it on `test.csv`, and redraws the graphs
   with the test results in the title.

### Expected warnings

When the model loads you will see a warning that the classifier weights are
newly initialized, and notes about unused masked-language-model weights. Both
are normal: the checkpoint was pretrained for masked-word prediction, and the
classification head is trained from scratch here.

### Outputs

Every run gets its own folder, so runs never overwrite each other:

```
outputs/
├── run_2026-09-30_14-05-12/
├── run_2026-09-30_16-40-03/
└── ...
```

The name is the local date and time the run started. If two runs start in the
same second, the second one gets `_2` appended. Each folder contains:

| File | Contents |
| --- | --- |
| `config.py` | Exact copy of the `config.py` used for this run |
| `best_model/` | Fine-tuned model and tokenizer (used by `predict.py`) |
| `history.json` | Training loss and validation loss, accuracy and macro-F1 per epoch |
| `training_curves.png` | Graphs of the training curves and metrics (see below) |
| `metrics.json` | Run name, best epoch, epochs run, training time, and test-set loss, accuracy and macro-F1 |
| `test_classification_report.txt` | Precision, recall and F1 for each of the 77 categories |

### Training-curves PNG

`training_curves.png` has two graphs side by side. Each graph's plotting area
is 4:3 (width:height).

- **Loss:** training and validation loss per epoch, on a log scale. Training
  loss falls by several orders of magnitude, which would flatten the
  validation curve on a linear axis.
- **Validation metrics:** validation accuracy and macro-F1 per epoch, in %.

Both graphs mark the best epoch with a dashed vertical line. The title shows
the run name and, once training is finished, the test accuracy, macro-F1 and
loss. The file is redrawn after every epoch, so an interrupted run still has
its graphs. If `VALIDATION_SPLIT = 0`, only training loss is plotted.

Graph size and resolution are set by `PLOT_GRAPH_WIDTH_INCHES` and `PLOT_DPI`
in `config.py`.

### Comparing and repeating runs

- **Compare:** open the `training_curves.png` and `metrics.json` of each run
  folder side by side.
- **Repeat a run:** copy its config back and train again. This starts a new
  run folder; the old one is left untouched.

  ```bash
  cp config.py config.py.bak                          # keep your current settings
  cp outputs/run_2026-09-30_14-05-12/config.py config.py
  python train.py
  ```

  With the same config, seed and package versions, results should be very
  close, though CPU arithmetic can differ slightly between machines and
  thread counts.

## Checking predictions

```bash
python predict.py
```

Uses the newest run in `outputs/` that has a saved model, and prints which run
it loaded. It runs 10 built-in banking queries. For each one it prints the
expected category, the predicted category with its probability, a ✓ or ✗, and
the top 3 predictions, then a final score such as `9/10 predictions match`.

To try your own sentences:

```bash
python predict.py "I lost my card" "Where is my money?"
```

To use a specific run, pass its folder name or path with `--run`:

```bash
python predict.py --run run_2026-09-30_14-05-12
python predict.py --run outputs/run_2026-09-30_14-05-12 "I lost my card"
```

`predict.py` reads `MAX_LENGTH` and `ATTN_IMPLEMENTATION` from the run's own
`config.py` copy, so it tokenizes the same way the model was trained.

Runs made before per-run folders existed (for example
`outputs/ettin-17m-banking77/`) are not picked automatically, but still work
with `--run outputs/ettin-17m-banking77`.

## Configuration

All settings live in `config.py`. The ones you are most likely to change:

| Setting | Default | Notes |
| --- | --- | --- |
| `NUM_EPOCHS` | `8` | Upper limit; early stopping may end sooner. |
| `LEARNING_RATE` | `8e-5` | |
| `TRAIN_BATCH_SIZE` | `32` | |
| `MAX_LENGTH` | `128` | Raise it if `train.py` reports truncated queries. |
| `VALIDATION_SPLIT` | `0.1` | Set to `0` to train on all of `train.csv` and keep the last epoch. |
| `EARLY_STOPPING_PATIENCE` | `3` | `None` disables early stopping. |
| `METRIC_FOR_BEST_MODEL` | `"accuracy"` | Or `"macro_f1"`. |
| `CLASSIFIER_DROPOUT` | `0.1` | The checkpoint's own default is `0.0`; set `None` to use it. |
| `CLASSIFIER_POOLING` | `"mean"` | Or `"cls"`. |
| `NUM_THREADS` | `None` | `None` lets PyTorch use all cores. |
| `SEED` | `42` | |
| `OUTPUTS_DIR` | `outputs/` | Where run folders are created. |
| `RUN_NAME_PREFIX` | `"run"` | Start of each run folder's name. |
| `PLOT_GRAPH_WIDTH_INCHES` | `6.4` | Width of one graph; its height is 3/4 of it. |
| `PLOT_DPI` | `150` | Resolution of the PNG. |
| `PREDICT_TOP_K` | `3` | Predictions shown per sentence in `predict.py`. |

Paths are resolved relative to `config.py`, so the scripts work from any
working directory.

## Troubleshooting

- **`No matching distribution found for torch==2.11.0+cpu`**: check that your
  Python is 3.10–3.14 on x86-64 Linux (`python --version`). To use a newer
  torch, pick an `X.Y.Z+cpu` version listed for your Python at
  <https://download.pytorch.org/whl/cpu/torch/> and change the pin.
- **`Due to a serious vulnerability issue in torch.load…`**: an older torch is
  active. Make sure the virtual environment is activated and reinstall.
- **`label value(s) … not in categories.json`**: a category in a CSV file is
  spelled differently from `categories.json`. Names must match exactly,
  including `Refund_not_showing_up` (capital R) and `reverted_card_payment?`
  (with the question mark).
- **`No trained run found`** from `predict.py`: run `python train.py` first,
  or point to a folder with `--run`.
- **`ModuleNotFoundError: No module named 'matplotlib'`**: the environment
  predates this version; run `pip install -r requirements.txt` again.
- **`[plot] WARNING: could not save training curves`**: drawing the graphs
  failed, but training carried on and the model, history and metrics are
  still saved. The message says what went wrong.
- **Training is slow**: set `NUM_THREADS` to your number of physical cores, or
  lower `NUM_EPOCHS`.
