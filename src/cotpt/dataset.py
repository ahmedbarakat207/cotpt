"""Real-data loading for training, as an alternative to the built-in toy
corpus in data.py. Nothing here was testable against a real HF `datasets`
download in the sandbox this was built in (no Hub access) -- the local-file
path is what was actually exercised; the `datasets` path is written
defensively (guarded import, clear error if missing) but wasn't run against
a real dataset. Try it against something small first.
"""

import os


def load_text_file(path: str, min_chars: int = 20):
    """Splits a plain-text file into training examples on blank lines
    (paragraph breaks). Filters out fragments too short to be useful."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    paragraphs = [p.strip() for p in raw.split("\n\n")]
    texts = [p for p in paragraphs if len(p) >= min_chars]
    if not texts:
        raise ValueError(
            f"No usable paragraphs (>= {min_chars} chars) found in {path}. "
            "Check the file isn't empty and paragraphs are separated by blank lines."
        )
    return texts


def load_hf_dataset(name: str, split: str = "train", text_field: str = "text", max_examples: int = None):
    """Loads a Hugging Face `datasets` dataset and pulls out a text field.
    Requires internet access and `pip install datasets` -- neither available
    in the sandbox this project was built in, so this path is untested; the
    local-file path (load_text_file) is the one that's actually been run.
    """
    try:
        import datasets as hf_datasets
    except ImportError as e:
        raise ImportError(
            "The `datasets` library isn't installed. Run `pip install datasets`, "
            "or use --data-path with a local .txt file instead."
        ) from e

    ds = hf_datasets.load_dataset(name, split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    if text_field not in ds.column_names:
        raise ValueError(f"'{text_field}' not found in dataset columns: {ds.column_names}")
    return [row[text_field] for row in ds if row[text_field] and len(row[text_field].strip()) > 0]


def load_training_texts(data_path: str = None, hf_dataset: str = None, hf_split: str = "train",
                         hf_text_field: str = "text", hf_max_examples: int = None):
    """Single entry point used by scripts/train.py: local file takes
    priority over an HF dataset name; if neither is given, the caller should
    fall back to cotpt.data.TRAIN_TEXTS itself."""
    if data_path:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"--data-path {data_path} does not exist")
        return load_text_file(data_path)
    if hf_dataset:
        return load_hf_dataset(hf_dataset, split=hf_split, text_field=hf_text_field, max_examples=hf_max_examples)
    return None
