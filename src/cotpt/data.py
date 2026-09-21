import json
import os
import re
from typing import List, Optional


DEFAULT_PROMPT = (
    "Q: Natalia sold clips to 48 of her friends in April, and then she sold "
    "half as many clips in May. How many clips did Natalia sell altogether "
    "in April and May?\nA:"
)


def load_jsonl(path: str, limit: Optional[int] = None) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def format_gsm8k(row: dict, mode: str = "step_by_step") -> str:
    q = row["question"].strip()
    a = row["answer"].strip()
    if mode == "direct_answer":
        target = a.split("####")[-1].strip() if "####" in a else a
        return f"Q: {q}\nA: The answer is {target}."
    clean = re.sub(r"<<.*?>>", "", a).strip()
    return f"Q: {q}\nA: {clean}"


def load_gsm8k(
    split: str = "train",
    mode: str = "step_by_step",
    limit: Optional[int] = None,
) -> List[str]:
    local_path = f"data/gsm8k_{split}.jsonl"
    if os.path.exists(local_path):
        return [format_gsm8k(row, mode) for row in load_jsonl(local_path, limit)]

    import datasets as hf_datasets
    ds = hf_datasets.load_dataset("openai/gsm8k", "main", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [format_gsm8k(row, mode) for row in ds]


def format_math(row: dict, mode: str = "step_by_step") -> str:
    p = row["problem"].strip()
    s = row["solution"].strip()
    if mode == "direct_answer":
        match = re.search(r"\\boxed\{([^}]+)\}", s)
        target = match.group(1).strip() if match else s
        return f"Q: {p}\nA: The answer is {target}."
    return f"Q: {p}\nA: {s}"


def load_math(
    split: str = "train",
    mode: str = "step_by_step",
    limit: Optional[int] = None,
    subject: str = "algebra",
) -> List[str]:
    import datasets as hf_datasets
    ds = hf_datasets.load_dataset("EleutherAI/hendrycks_math", subject, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [format_math(row, mode) for row in ds]


def format_mmlu(row: dict) -> str:
    choices = "\n".join([f"({chr(65 + i)}) {c}" for i, c in enumerate(row["choices"])])
    target = chr(65 + row["answer"])
    return f"Question: {row['question'].strip()}\n{choices}\nAnswer: ({target})"


def load_mmlu(
    split: str = "test",
    limit: Optional[int] = None,
    subject: str = "college_mathematics",
) -> List[str]:
    import datasets as hf_datasets
    ds = hf_datasets.load_dataset("cais/mmlu", subject, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [format_mmlu(row) for row in ds]


def format_arc(row: dict) -> str:
    choices_text = row["choices"]["text"]
    choices_label = row["choices"]["label"]
    choices = "\n".join([f"({lbl}) {txt}" for lbl, txt in zip(choices_label, choices_text)])
    target = row["answerKey"].strip()
    return f"Question: {row['question'].strip()}\n{choices}\nAnswer: ({target})"


def load_arc(
    split: str = "test",
    limit: Optional[int] = None,
    subset: str = "ARC-Challenge",
) -> List[str]:
    import datasets as hf_datasets
    ds = hf_datasets.load_dataset("allenai/ai2_arc", subset, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [format_arc(row) for row in ds]


def format_svamp(row: dict) -> str:
    body = row.get("Body", "").strip()
    question = row.get("Question", "").strip()
    full_q = f"{body} {question}".strip()
    ans = str(row["Answer"]).strip()
    return f"Q: {full_q}\nA: The answer is {ans}."


def load_svamp(
    split: str = "train",
    limit: Optional[int] = None,
) -> List[str]:
    import datasets as hf_datasets
    ds = hf_datasets.load_dataset("ChilleD/SVAMP", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [format_svamp(row) for row in ds]


def load_fineweb_edu(limit: Optional[int] = None) -> List[str]:
    path = "data/fineweb_edu_train.jsonl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    texts = []
    for row in load_jsonl(path, limit):
        text = row.get("text", "").strip()
        if len(text) >= 100:
            texts.append(text)
    return texts


def load_text_file(path: str, min_chars: int = 20) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".jsonl"):
        texts = []
        for row in load_jsonl(path):
            if "question" in row and "answer" in row:
                item = format_gsm8k(row)
            elif "problem" in row and "solution" in row:
                item = format_math(row)
            elif "text" in row:
                item = row["text"].strip()
            else:
                continue
            if len(item) >= min_chars:
                texts.append(item)
        if not texts:
            raise ValueError(f"No usable examples in {path}.")
        return texts
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    paragraphs = [p.strip() for p in raw.split("\n\n")]
    texts = [p for p in paragraphs if len(p) >= min_chars]
    if not texts:
        raise ValueError(f"No usable paragraphs in {path}.")
    return texts


def load_hf_dataset(
    name: str,
    split: str = "train",
    text_field: str = "text",
    max_examples: Optional[int] = None,
) -> List[str]:
    import datasets as hf_datasets
    clean = name.strip().lower()
    if clean in ("gsm8k", "openai/gsm8k"):
        return load_gsm8k(split=split, limit=max_examples)
    if clean in ("math", "hendrycks_math", "eleutherai/hendrycks_math"):
        return load_math(split=split, limit=max_examples)
    if clean in ("mmlu", "cais/mmlu"):
        return load_mmlu(split=split, limit=max_examples)
    if clean in ("arc", "ai2_arc", "allenai/ai2_arc"):
        return load_arc(split=split, limit=max_examples)
    if clean in ("svamp", "chilled/svamp"):
        return load_svamp(split=split, limit=max_examples)

    ds = hf_datasets.load_dataset(name.strip(), split=split)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    if text_field in ds.column_names:
        return [r[text_field].strip() for r in ds if r[text_field] and r[text_field].strip()]
    if "question" in ds.column_names and "answer" in ds.column_names:
        return [f"Q: {r['question'].strip()}\nA: {r['answer'].strip()}" for r in ds]
    raise ValueError(f"'{text_field}' not in columns: {ds.column_names}")


def find_prompt_boundary(text: str, tokenizer) -> Optional[int]:
    delimiters = ["\nA:", "\nAnswer:"]
    for delim in delimiters:
        if delim in text:
            prefix = text.split(delim)[0] + delim
            return max(0, tokenizer(prefix, return_tensors="pt").input_ids.shape[1] - 1)
    return None


def load_training_texts(
    data_path: Optional[str] = None,
    hf_dataset: Optional[str] = None,
    hf_split: str = "train",
    hf_text_field: str = "text",
    hf_max_examples: Optional[int] = None,
    dataset_name: Optional[str] = None,
    mode: str = "step_by_step",
) -> Optional[List[str]]:
    if data_path:
        return load_text_file(data_path)
    if hf_dataset:
        return load_hf_dataset(hf_dataset, split=hf_split, text_field=hf_text_field, max_examples=hf_max_examples)
    if dataset_name:
        name = dataset_name.lower().strip()
        if name in ("gsm8k", "gsm8k_train"):
            return load_gsm8k(split=hf_split, mode=mode, limit=hf_max_examples)
        if name in ("gsm8k_direct", "direct_math"):
            return load_gsm8k(split=hf_split, mode="direct_answer", limit=hf_max_examples)
        if name in ("math", "hendrycks_math"):
            return load_math(split=hf_split, mode=mode, limit=hf_max_examples)
        if name in ("math_direct",):
            return load_math(split=hf_split, mode="direct_answer", limit=hf_max_examples)
        if name in ("mmlu", "cais/mmlu"):
            return load_mmlu(split="test" if hf_split == "test" else "dev", limit=hf_max_examples)
        if name in ("arc", "arc_challenge", "ai2_arc"):
            return load_arc(split="train" if hf_split == "train" else "test", limit=hf_max_examples)
        if name in ("svamp", "chilled/svamp"):
            return load_svamp(split="train" if hf_split == "train" else "test", limit=hf_max_examples)
        if name in ("fineweb_edu", "fineweb"):
            return load_fineweb_edu(limit=hf_max_examples)
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return None
