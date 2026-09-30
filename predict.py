"""
Check a fine-tuned banking77 model by hand.

Usage:
    python predict.py                                   # 10 built-in sentences, newest run
    python predict.py "I lost my card" "Where is my money?"   # your own sentences
    python predict.py --run run_2026-09-30_14-05-12      # a specific run (name or path)
    python predict.py --run outputs/run_2026-09-30_14-05-12 "I lost my card"

By default the newest run folder in outputs/ that contains a saved model is
used. For each sentence the script prints the top-k predicted categories with
their probabilities. For the built-in sentences it also shows the category a
human would expect, so you can compare them at a glance.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config

# (sentence, expected banking77 category)
EXAMPLES = [
    ("I ordered my card two weeks ago and it still hasn't arrived.", "card_arrival"),
    ("How do I activate the card you just sent me?", "activate_my_card"),
    ("My card was declined when I tried to pay at the supermarket.", "declined_card_payment"),
    ("The ATM kept my card and didn't give it back.", "card_swallowed"),
    ("Why was I charged a fee for taking cash out of the ATM?", "cash_withdrawal_charge"),
    ("I lost my phone, what should I do to protect my account?", "lost_or_stolen_phone"),
    ("How can I change the PIN on my card?", "change_pin"),
    ("I was charged twice for the same purchase.", "transaction_charged_twice"),
    ("Can I use Apple Pay with my card?", "apple_pay_or_google_pay"),
    ("Please close my account, I no longer need it.", "terminate_account"),
]


# --------------------------------------------------------------------------- #
# Choosing the run
# --------------------------------------------------------------------------- #
def has_model(run_dir: Path) -> bool:
    return (run_dir / config.BEST_MODEL_DIRNAME).is_dir()


def find_latest_run() -> Path:
    """Newest outputs/<prefix>_<timestamp> folder that has a saved model.
    The timestamp format sorts chronologically, so the last name is the newest."""
    outputs_dir = Path(config.OUTPUTS_DIR)
    prefix = f"{config.RUN_NAME_PREFIX}_"
    runs = sorted(
        d for d in outputs_dir.glob(f"{prefix}*") if d.is_dir() and has_model(d)
    ) if outputs_dir.is_dir() else []
    if not runs:
        sys.exit(f"No trained run found in {outputs_dir}. Run `python train.py` first, "
                 f"or pass --run <folder>.")
    return runs[-1]


def resolve_run(arg: str) -> Path:
    """Accept a path to a run folder, or just its name inside outputs/."""
    candidate = Path(arg).expanduser()
    if not candidate.is_dir():
        candidate = Path(config.OUTPUTS_DIR) / arg
    if not candidate.is_dir():
        sys.exit(f"Run folder not found: {arg}")
    if not has_model(candidate):
        sys.exit(f"No {config.BEST_MODEL_DIRNAME}/ folder in {candidate}.")
    return candidate.resolve()


def load_run_settings(run_dir: Path) -> dict:
    """Use the tokenisation settings the run was trained with, taken from its
    config.py copy. Falls back to the current config.py (e.g. for runs made
    before config copies existed)."""
    settings = {
        "MAX_LENGTH": config.MAX_LENGTH,
        "ATTN_IMPLEMENTATION": config.ATTN_IMPLEMENTATION,
    }
    copy_path = run_dir / config.CONFIG_COPY_FILENAME
    if copy_path.is_file():
        try:
            spec = importlib.util.spec_from_file_location("run_config", copy_path)
            run_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run_config)
            for key in settings:
                if hasattr(run_config, key):
                    settings[key] = getattr(run_config, key)
        except Exception as exc:  # noqa: BLE001
            print(f"[predict] could not read {copy_path.name} ({exc}); using current config.py")
    return settings


# --------------------------------------------------------------------------- #
# Model and prediction
# --------------------------------------------------------------------------- #
def load_model(model_dir: Path, attn_implementation):
    kwargs = {}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, **kwargs)
    model.to(config.DEVICE).eval()
    return tokenizer, model


@torch.inference_mode()
def predict(sentences, tokenizer, model, top_k: int, max_length: int):
    enc = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_token_type_ids=False,
        return_tensors="pt",
    ).to(config.DEVICE)
    probs = model(**enc).logits.softmax(dim=-1)
    top = probs.topk(min(top_k, probs.size(-1)), dim=-1)
    id2label = model.config.id2label
    return [
        [(id2label[int(i)], float(p)) for p, i in zip(values, indices)]
        for values, indices in zip(top.values.tolist(), top.indices.tolist())
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Check a fine-tuned banking77 model.")
    parser.add_argument("sentences", nargs="*",
                        help="sentences to classify (default: 10 built-in examples)")
    parser.add_argument("--run", default=None,
                        help="run folder name or path (default: newest run in outputs/)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if config.NUM_THREADS:
        torch.set_num_threads(config.NUM_THREADS)

    run_dir = resolve_run(args.run) if args.run else find_latest_run()
    settings = load_run_settings(run_dir)
    print(f"[predict] run: {run_dir.name}  ({run_dir})")

    examples = [(s, None) for s in args.sentences] if args.sentences else EXAMPLES
    tokenizer, model = load_model(run_dir / config.BEST_MODEL_DIRNAME,
                                  settings["ATTN_IMPLEMENTATION"])

    # Catch typos in the expected labels: they would always look "wrong".
    known = set(model.config.id2label.values())
    unknown = sorted({exp for _, exp in examples if exp is not None and exp not in known})
    if unknown:
        sys.exit(f"Expected label(s) not in the model's categories: {unknown}")

    results = predict([s for s, _ in examples], tokenizer, model,
                      config.PREDICT_TOP_K, settings["MAX_LENGTH"])

    n_checked, n_correct = 0, 0
    for idx, ((sentence, expected), top) in enumerate(zip(examples, results), start=1):
        best_label, best_prob = top[0]
        print(f"\n[{idx}] {sentence}")
        if expected is not None:
            n_checked += 1
            correct = best_label == expected
            n_correct += correct
            print(f"    expected : {expected}")
            print(f"    predicted: {best_label} ({best_prob:.1%})  {'✓' if correct else '✗'}")
        else:
            print(f"    predicted: {best_label} ({best_prob:.1%})")
        print("    top-{}    : {}".format(
            len(top), " | ".join(f"{label} {prob:.1%}" for label, prob in top)))

    if n_checked:
        print(f"\n{n_correct}/{n_checked} predictions match the expected category.")


if __name__ == "__main__":
    main()
