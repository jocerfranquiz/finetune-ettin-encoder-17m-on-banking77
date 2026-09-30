"""
Fine-tune the local `ettin-encoder-17m` checkpoint on banking77 (77-way intent
classification) on CPU, with a plain PyTorch training loop.

Usage:
    python train.py

All settings (paths, hyperparameters, runtime options) are in config.py.

Each run writes to its own folder, outputs/run_<date>_<time>/, containing:
    config.py                       exact copy of the config used (for re-runs)
    best_model/                     fine-tuned model + tokenizer
    history.json                    train loss and validation metrics per epoch
    training_curves.png             loss and metric graphs (each graph 4:3)
    metrics.json                    final test-set metrics of the best model
    test_classification_report.txt  per-class precision / recall / F1 on test
"""

import json
import math
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")  # render to file only; no display needed
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

import config


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def check_config() -> None:
    required = [config.TRAIN_FILE, config.TEST_FILE, config.CATEGORIES_FILE, config.MODEL_DIR]
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required path(s):\n  " + "\n  ".join(missing))
    if config.METRIC_FOR_BEST_MODEL not in ("accuracy", "macro_f1"):
        raise ValueError("config.METRIC_FOR_BEST_MODEL must be 'accuracy' or 'macro_f1'.")
    if config.LR_SCHEDULER not in ("linear", "cosine"):
        raise ValueError("config.LR_SCHEDULER must be 'linear' or 'cosine'.")
    if not 0 <= config.VALIDATION_SPLIT < 1:
        raise ValueError("config.VALIDATION_SPLIT must be in [0, 1).")


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Run folder
# --------------------------------------------------------------------------- #
def create_run_dir() -> Path:
    """Create outputs/<prefix>_<timestamp>/ ; add _2, _3... if the name is taken."""
    stamp = datetime.now().strftime(config.RUN_TIMESTAMP_FORMAT)
    base_name = f"{config.RUN_NAME_PREFIX}_{stamp}"
    outputs_dir = Path(config.OUTPUTS_DIR)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    run_dir, n = outputs_dir / base_name, 1
    while True:
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            n += 1
            run_dir = outputs_dir / f"{base_name}_{n}"


def run_paths(run_dir: Path) -> SimpleNamespace:
    """All output file locations for one run."""
    return SimpleNamespace(
        run_dir=run_dir,
        best_model=run_dir / config.BEST_MODEL_DIRNAME,
        history=run_dir / config.HISTORY_FILENAME,
        metrics=run_dir / config.METRICS_FILENAME,
        test_report=run_dir / config.TEST_REPORT_FILENAME,
        plot=run_dir / config.PLOT_FILENAME,
        config_copy=run_dir / config.CONFIG_COPY_FILENAME,
    )


def copy_config(destination: Path) -> None:
    """Save an exact copy of config.py so the run can be repeated.
    Done at the start, so even an interrupted run keeps its settings."""
    shutil.copy2(Path(config.__file__).resolve(), destination)


# --------------------------------------------------------------------------- #
# Training-curves plot
# --------------------------------------------------------------------------- #
# Colours: slots 1-2 of a colourblind-checked categorical palette, plus
# neutral inks for text, grid and axes.
_SERIES_1 = "#2a78d6"   # blue
_SERIES_2 = "#eb6834"   # orange
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_SURFACE = "#fcfcfb"


def _style_axes(ax, title: str, ylabel: str) -> None:
    ax.set_box_aspect(3 / 4)  # plotting area is exactly 4:3 (width:height)
    ax.set_facecolor(_SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=_INK, pad=10)
    ax.set_xlabel("Epoch", color=_INK_2)
    ax.set_ylabel(ylabel, color=_INK_2)
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.tick_params(colors=_INK_2, labelsize=9)


def _plot_series(ax, x, y, label: str, color: str) -> None:
    ax.plot(x, y, label=label, color=color, linewidth=2, solid_joinstyle="round",
            solid_capstyle="round", marker="o", markersize=6,
            markeredgecolor=_SURFACE, markeredgewidth=1.5)


def _mark_best_epoch(ax, best_epoch) -> None:
    if best_epoch is None:
        return
    ax.axvline(best_epoch, color=_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(f"best epoch {best_epoch}", xy=(best_epoch, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -4), textcoords="offset points", ha="left", va="top",
                fontsize=8.5, color=_INK_2)


def plot_training_curves(history, path: Path, best_epoch=None, test_metrics=None,
                         run_name: str = "") -> None:
    """Save a PNG with two graphs, each with a 4:3 plotting area:
    (1) training and validation loss (log scale), (2) validation accuracy and
    macro-F1. The best epoch is marked, and test results (if given) are shown
    in the title."""
    if not history:
        return
    epochs = [r["epoch"] for r in history]
    has_val = "val_loss" in history[0]

    width = float(config.PLOT_GRAPH_WIDTH_INCHES)
    fig, (ax_loss, ax_metric) = plt.subplots(
        1, 2, figsize=(2 * width, width * 3 / 4 + 1.2), layout="constrained",
        facecolor="white",
    )

    # (1) Loss. Log scale: training loss falls by several orders of magnitude,
    # which would flatten the validation curve on a linear axis.
    _style_axes(ax_loss, "Loss", "Cross-entropy (log scale)")
    _plot_series(ax_loss, epochs, [r["train_loss"] for r in history], "Train", _SERIES_1)
    if has_val:
        _plot_series(ax_loss, epochs, [r["val_loss"] for r in history], "Validation", _SERIES_2)
    ax_loss.set_yscale("log")
    _mark_best_epoch(ax_loss, best_epoch)
    ax_loss.legend(frameon=False, labelcolor=_INK_2, loc="upper right")

    # (2) Validation metrics, in %.
    _style_axes(ax_metric, "Validation metrics", "Score (%)")
    if has_val:
        _plot_series(ax_metric, epochs, [100 * r["val_accuracy"] for r in history],
                     "Accuracy", _SERIES_1)
        _plot_series(ax_metric, epochs, [100 * r["val_macro_f1"] for r in history],
                     "Macro-F1", _SERIES_2)
        _mark_best_epoch(ax_metric, best_epoch)
        ax_metric.legend(frameon=False, labelcolor=_INK_2, loc="lower right")
    else:
        ax_metric.text(0.5, 0.5, "No validation split\n(VALIDATION_SPLIT = 0)",
                       transform=ax_metric.transAxes, ha="center", va="center",
                       color=_MUTED, fontsize=11)
        ax_metric.set_yticks([])

    for ax in (ax_loss, ax_metric):
        if len(epochs) <= 20:          # one tick per epoch; otherwise automatic
            ax.set_xticks(epochs)
        ax.set_xlim(min(epochs) - 0.3, max(epochs) + 0.3)

    title = f"Training curves — {run_name}" if run_name else "Training curves"
    if test_metrics:
        title += (f"\nTest: accuracy {100 * test_metrics['accuracy']:.2f}%  ·  "
                  f"macro-F1 {100 * test_metrics['macro_f1']:.2f}%  ·  "
                  f"loss {test_metrics['loss']:.4f}")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=_INK)

    fig.savefig(path, dpi=config.PLOT_DPI, facecolor="white")
    plt.close(fig)


def save_plot_safely(*args, **kwargs) -> None:
    """A plotting problem must never stop or lose a training run."""
    try:
        plot_training_curves(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[plot] WARNING: could not save training curves: {exc}")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_label_names(path: Path) -> list:
    """Read categories.json. Accepts a list of names, {name: id} or {id: name}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        names = [str(x).strip() for x in data]
    elif isinstance(data, dict):
        values = list(data.values())
        if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            # {name: id}
            names = [str(k).strip() for k, _ in sorted(data.items(), key=lambda kv: kv[1])]
        elif all(str(k).strip().isdigit() for k in data):
            # {"0": name, "1": name, ...}
            names = [str(data[k]).strip() for k in sorted(data, key=lambda k: int(k))]
        else:
            raise ValueError(f"Unrecognised structure in {path}.")
    else:
        raise ValueError(f"Unrecognised structure in {path}.")

    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate category names in {path}.")
    return names


def resolve_column(df: pd.DataFrame, preferred: str, candidates, what: str, path: Path) -> str:
    if preferred in df.columns:
        return preferred
    for col in candidates:
        if col in df.columns:
            print(f"[data] column '{preferred}' not in {path.name}; using '{col}' as {what} column")
            return col
    raise KeyError(
        f"No {what} column found in {path}. Columns are {list(df.columns)}. "
        f"Set the right name in config.py."
    )


def encode_labels(raw_values, label2id: dict, num_labels: int, path: Path) -> list:
    """Map label values (category names or integer ids) to integer ids."""
    ids, unknown = [], set()
    for value in raw_values:
        key = str(value).strip()
        if key in label2id:
            ids.append(label2id[key])
            continue
        try:
            number = float(key)
        except ValueError:
            number = None
        if number is not None and number.is_integer() and 0 <= int(number) < num_labels:
            ids.append(int(number))
        else:
            unknown.add(key)
    if unknown:
        preview = sorted(unknown)[:10]
        raise ValueError(f"{len(unknown)} label value(s) in {path} not in categories.json, e.g. {preview}")
    return ids


def load_split(path: Path, label2id: dict, num_labels: int):
    # dtype=str + keep_default_na=False: read every cell as the exact text in
    # the file. Otherwise pandas turns strings such as "NA", "None" or "null"
    # into missing values and silently drops those rows.
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    text_col = resolve_column(df, config.TEXT_COLUMN, config.TEXT_COLUMN_CANDIDATES, "text", path)
    label_col = resolve_column(df, config.LABEL_COLUMN, config.LABEL_COLUMN_CANDIDATES, "label", path)

    df = df[[text_col, label_col]].dropna()
    texts = df[text_col].astype(str).str.strip()
    keep = texts.str.len() > 0
    texts = texts[keep].tolist()
    labels = encode_labels(df.loc[keep, label_col].tolist(), label2id, num_labels, path)
    return texts, labels


def report_label_coverage(labels, label_names, split_name: str) -> None:
    """Warn if some categories never appear in a split."""
    present = set(labels)
    missing = [name for i, name in enumerate(label_names) if i not in present]
    if missing:
        print(f"[data] WARNING: {len(missing)} categories have no examples in {split_name}: {missing[:10]}")
    else:
        print(f"[data] all {len(label_names)} categories present in {split_name}")


def report_truncation(texts, tokenizer, max_length: int, split_name: str) -> None:
    """Count queries longer than max_length tokens (they get truncated)."""
    lengths = [len(ids) for ids in tokenizer(texts, add_special_tokens=True)["input_ids"]]
    n_truncated = sum(length > max_length for length in lengths)
    print(f"[data] {split_name}: longest query = {max(lengths)} tokens, "
          f"{n_truncated} truncated at MAX_LENGTH={max_length}")


class IntentDataset(Dataset):
    """Pre-tokenised examples; padding is done per batch by the collator."""

    def __init__(self, texts, labels, tokenizer, max_length: int):
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            return_token_type_ids=False,
        )
        self.items = [
            {
                "input_ids": enc["input_ids"][i],
                "attention_mask": enc["attention_mask"][i],
                "labels": labels[i],
            }
            for i in range(len(texts))
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# --------------------------------------------------------------------------- #
# Model / optimisation
# --------------------------------------------------------------------------- #
def model_load_kwargs() -> dict:
    kwargs = {}
    if config.ATTN_IMPLEMENTATION:
        kwargs["attn_implementation"] = config.ATTN_IMPLEMENTATION
    return kwargs


def build_model(num_labels: int, id2label: dict, label2id: dict):
    kwargs = model_load_kwargs()
    kwargs.update(num_labels=num_labels, id2label=id2label, label2id=label2id)
    if config.CLASSIFIER_POOLING is not None:
        kwargs["classifier_pooling"] = config.CLASSIFIER_POOLING
    if config.CLASSIFIER_DROPOUT is not None:
        kwargs["classifier_dropout"] = config.CLASSIFIER_DROPOUT
    # The checkpoint is a masked-LM encoder: a warning that the classification
    # head is newly initialised is expected here.
    return AutoModelForSequenceClassification.from_pretrained(config.MODEL_DIR, **kwargs)


def build_optimizer(model) -> torch.optim.Optimizer:
    # No weight decay on biases and normalisation weights (1-D parameters).
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim < 2 else decay).append(param)
    groups = [
        {"params": decay, "weight_decay": config.WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=config.LEARNING_RATE,
        betas=tuple(config.ADAM_BETAS),
        eps=config.ADAM_EPSILON,
    )


def build_scheduler(optimizer, total_steps: int):
    warmup_steps = int(round(config.WARMUP_RATIO * total_steps))
    if config.LR_SCHEDULER == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)


def to_device(batch, device):
    batch = dict(batch)
    labels = batch.pop("labels").to(device)
    inputs = {k: v.to(device) for k, v in batch.items()}
    return inputs, labels


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_n = 0.0, 0
    preds, golds = [], []
    for batch in loader:
        inputs, labels = to_device(batch, device)
        logits = model(**inputs).logits
        total_loss += criterion(logits, labels).item() * labels.size(0)
        total_n += labels.size(0)
        preds.extend(logits.argmax(dim=-1).tolist())
        golds.extend(labels.tolist())
    metrics = {
        "loss": total_loss / max(total_n, 1),
        "accuracy": accuracy_score(golds, preds),
        "macro_f1": f1_score(golds, preds, average="macro", zero_division=0),
    }
    return metrics, preds, golds


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    check_config()
    set_seed(config.SEED)
    if config.NUM_THREADS:
        torch.set_num_threads(config.NUM_THREADS)
    device = torch.device(config.DEVICE)

    # ---- run folder ----
    paths = run_paths(create_run_dir())
    run_name = paths.run_dir.name
    copy_config(paths.config_copy)
    print(f"[run] {run_name}  ->  {paths.run_dir}")

    # ---- labels & data ----
    label_names = load_label_names(config.CATEGORIES_FILE)
    num_labels = len(label_names)
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for i, name in enumerate(label_names)}
    print(f"[data] {num_labels} categories")

    train_texts, train_labels = load_split(config.TRAIN_FILE, label2id, num_labels)
    test_texts, test_labels = load_split(config.TEST_FILE, label2id, num_labels)
    report_label_coverage(train_labels, label_names, "train.csv")
    report_label_coverage(test_labels, label_names, "test.csv")

    if config.VALIDATION_SPLIT > 0:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts,
            train_labels,
            test_size=config.VALIDATION_SPLIT,
            stratify=train_labels,
            random_state=config.SEED,
        )
    else:
        val_texts, val_labels = [], []
    print(f"[data] train={len(train_texts)}  val={len(val_texts)}  test={len(test_texts)}")

    # ---- tokenizer & model ----
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_DIR)
    report_truncation(train_texts + val_texts, tokenizer, config.MAX_LENGTH, "train.csv")
    report_truncation(test_texts, tokenizer, config.MAX_LENGTH, "test.csv")
    model = build_model(num_labels, id2label, label2id).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {model.__class__.__name__}  params={n_params / 1e6:.1f}M  device={device}  "
          f"threads={torch.get_num_threads()}")

    collator = DataCollatorWithPadding(tokenizer)
    generator = torch.Generator().manual_seed(config.SEED)
    train_loader = DataLoader(
        IntentDataset(train_texts, train_labels, tokenizer, config.MAX_LENGTH),
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.NUM_WORKERS,
        generator=generator,
    )
    val_loader = None
    if val_texts:
        val_loader = DataLoader(
            IntentDataset(val_texts, val_labels, tokenizer, config.MAX_LENGTH),
            batch_size=config.EVAL_BATCH_SIZE,
            shuffle=False,
            collate_fn=collator,
            num_workers=config.NUM_WORKERS,
        )
    test_loader = DataLoader(
        IntentDataset(test_texts, test_labels, tokenizer, config.MAX_LENGTH),
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.NUM_WORKERS,
    )

    # ---- optimisation ----
    total_steps = len(train_loader) * config.NUM_EPOCHS
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, total_steps)
    train_criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    eval_criterion = nn.CrossEntropyLoss()
    print(f"[train] epochs={config.NUM_EPOCHS}  steps/epoch={len(train_loader)}  total_steps={total_steps}")

    best_score = -math.inf
    best_epoch = None
    epochs_without_improvement = 0
    history = []
    global_step = 0
    start = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        epoch_loss, epoch_n = 0.0, 0
        window_loss, window_n = 0.0, 0

        for batch in train_loader:
            inputs, labels = to_device(batch, device)
            logits = model(**inputs).logits
            loss = train_criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.MAX_GRAD_NORM:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            global_step += 1

            bs = labels.size(0)
            epoch_loss += loss.item() * bs
            epoch_n += bs
            window_loss += loss.item() * bs
            window_n += bs

            if global_step % config.LOG_EVERY_N_STEPS == 0:
                print(f"  epoch {epoch}  step {global_step}/{total_steps}  "
                      f"loss={window_loss / window_n:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                      f"elapsed={(time.time() - start) / 60:.1f} min")
                window_loss, window_n = 0.0, 0

        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(epoch_n, 1),
            "elapsed_min": round((time.time() - start) / 60, 2),
        }

        if val_loader is not None:
            val_metrics, _, _ = evaluate(model, val_loader, eval_criterion, device)
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
            score = val_metrics[config.METRIC_FOR_BEST_MODEL]
            improved = score > best_score
            if improved:
                best_score, best_epoch = score, epoch
                epochs_without_improvement = 0
                model.save_pretrained(paths.best_model)
                tokenizer.save_pretrained(paths.best_model)
            else:
                epochs_without_improvement += 1
            print(f"[epoch {epoch}] train_loss={record['train_loss']:.4f}  "
                  f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
                  f"val_macro_f1={val_metrics['macro_f1']:.4f}"
                  + ("  <- new best, saved" if improved else ""))
        else:
            # No validation split: keep the latest epoch.
            best_epoch = epoch
            model.save_pretrained(paths.best_model)
            tokenizer.save_pretrained(paths.best_model)
            print(f"[epoch {epoch}] train_loss={record['train_loss']:.4f}  (saved)")

        history.append(record)
        write_json(paths.history, history)
        # Updated every epoch, so an interrupted run still has its graphs.
        save_plot_safely(history, paths.plot, best_epoch=best_epoch, run_name=run_name)

        if (
            val_loader is not None
            and config.EARLY_STOPPING_PATIENCE is not None
            and epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE
        ):
            print(f"[train] early stopping: no improvement for {epochs_without_improvement} epoch(s)")
            break

    train_minutes = (time.time() - start) / 60
    print(f"[train] done in {train_minutes:.1f} min; best epoch = {best_epoch}")

    # ---- final evaluation of the best checkpoint on the test set ----
    best_model = AutoModelForSequenceClassification.from_pretrained(
        paths.best_model, **model_load_kwargs()
    ).to(device)
    test_metrics, preds, golds = evaluate(best_model, test_loader, eval_criterion, device)
    print(f"[test] loss={test_metrics['loss']:.4f}  accuracy={test_metrics['accuracy']:.4f}  "
          f"macro_f1={test_metrics['macro_f1']:.4f}")

    report = classification_report(
        golds,
        preds,
        labels=list(range(num_labels)),
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
    paths.test_report.write_text(report, encoding="utf-8")

    write_json(
        paths.metrics,
        {
            "run_name": run_name,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "selection_metric": config.METRIC_FOR_BEST_MODEL if val_loader is not None else "last_epoch",
            "best_val_score": best_score if val_loader is not None else None,
            "train_minutes": round(train_minutes, 2),
            "test": test_metrics,
        },
    )
    # Final version of the graphs, with the test results in the title.
    save_plot_safely(history, paths.plot, best_epoch=best_epoch,
                     test_metrics=test_metrics, run_name=run_name)

    print(f"[done] run folder: {paths.run_dir}")
    print(f"[done] graphs: {paths.plot.name}  metrics: {paths.metrics.name}  "
          f"per-class report: {paths.test_report.name}  config copy: {paths.config_copy.name}")


if __name__ == "__main__":
    main()
