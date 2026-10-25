import os, re, ast, json, math, random, time, textwrap
from pathlib import Path

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments,
)


MODEL_NAME = "./bert-base-uncased"
PROJECT_DIR  = Path("project_code")
OUTPUT_DIR   = Path("lab1_model")
RESULTS_JSON = Path("lab1_results.json")
MAX_LEN      = 256
SEED         = 42


def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_project_files(project_dir: Path = PROJECT_DIR):
    if project_dir.exists():
        files = [p for p in project_dir.rglob("*.py") if p.is_file()]
    else:
        files = [
            p for p in Path(".").glob("*.py")
            if p.name not in ("train_lab1.py", "api_lab1.py")
        ]
    if not files:
        raise FileNotFoundError(
            f"Не найдены .py файлы в '{project_dir}' или текущей папке. "
        )
    return files


def extract_pairs(source_code: str):
    pairs = []
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"[AST] ошибка парсинга: {e}")
        return pairs

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node) or ""
            body_nodes = [
                b for b in node.body
                if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))
            ]
            if not body_nodes:
                continue
            body_src = "\n".join(
                textwrap.dedent(ast.get_source_segment(source_code, b) or "")
                for b in body_nodes
            ).strip()
            if not body_src:
                continue

            args = ", ".join(a.arg for a in node.args.args)
            sig = f"def {node.name}({args}):"
            if doc:
                pairs.append({"input": doc.strip(), "target": body_src,
                              "kind": "docstring->body"})
            pairs.append({"input": sig, "target": body_src,
                          "kind": "signature->body"})

        if isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node) or ""
            if doc:
                pairs.append({"input": f"class {node.name}:",
                              "target": doc.strip(),
                              "kind": "class->docstring"})
    return pairs


def read_corpus():
    files = find_project_files()
    print(f"[Corpus] Файлов: {len(files)}")
    for f in files:
        print(f"        - {f}")

    texts = []
    for f in files:
        src = clean_text(f.read_text(encoding="utf-8", errors="ignore"))
        if len(src) < 40:
            continue
        texts.append(src)
    return texts, files


def make_chunks(texts, max_lines=12, stride=6):
    chunks = []
    for text in texts:
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            continue
        for start in range(0, len(lines), stride):
            chunk = "\n".join(lines[start:start + max_lines])
            if len(chunk) >= 40:
                chunks.append(chunk)
            if start + max_lines >= len(lines):
                break
    unique = list(dict.fromkeys(chunks))
    print(f"[Chunks] всего: {len(unique)}")
    return unique


def split_texts(texts):
    random.seed(SEED)
    data = texts[:]
    random.shuffle(data)
    n = len(data)
    return data[: int(0.8 * n)], data[int(0.8 * n): int(0.9 * n)], data[int(0.9 * n):]


def tokenize_dataset(texts, tokenizer):
    ds = Dataset.from_dict({"text": texts})
    def tok(ex):
        return tokenizer(ex["text"], truncation=True, max_length=MAX_LEN)
    return ds.map(tok, batched=True, remove_columns=["text"])


def train_experiment(name, train_ds, val_ds, tokenizer, epochs, bs, lr):
    print(f"\n[Exp:{name}] epochs={epochs}, bs={bs}, lr={lr}")
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
    )
    args = TrainingArguments(
        output_dir=f"./runs/{name}",
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        learning_rate=lr,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        seed=SEED,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
    )
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    ev = trainer.evaluate()
    loss, ppl = ev["eval_loss"], math.exp(ev["eval_loss"])
    print(f"[Exp:{name}] loss={loss:.4f}  ppl={ppl:.2f}  time={dt:.1f}s")
    return {"name": name, "epochs": epochs, "batch_size": bs, "lr": lr,
            "eval_loss": loss, "perplexity": ppl,
            "train_time_sec": round(dt, 1), "trainer": trainer}


@torch.no_grad()
def masked_accuracy(model, tokenizer, texts, n_samples=100):
    model.eval()
    device = next(model.parameters()).device
    correct = total = 0
    for text in texts[:n_samples]:
        enc = tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=128)
        ids, attn = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        if ids.size(1) < 5:
            continue
        special = set(tokenizer.all_special_ids)
        cand = [i for i in range(1, ids.size(1) - 1)
                if ids[0, i].item() not in special]
        if not cand:
            continue
        k = max(1, int(0.15 * len(cand)))
        positions = random.sample(cand, k)
        masked = ids.clone()
        for p in positions:
            masked[0, p] = tokenizer.mask_token_id
        logits = model(input_ids=masked, attention_mask=attn).logits
        pred = logits.argmax(-1)
        for p in positions:
            total += 1
            correct += int(pred[0, p].item() == ids[0, p].item())
    return correct / total if total else 0.0


@torch.no_grad()
def generate_completion(model, tokenizer, prompt, max_new_tokens=8):
    """BERT — masked LM: добавляем [MASK] в конец, предсказываем, повторяем."""
    model.eval()
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt",
                    truncation=True, max_length=MAX_LEN - max_new_tokens)
    ids = enc["input_ids"].to(device)
    for _ in range(max_new_tokens):
        mask = torch.tensor([[tokenizer.mask_token_id]], device=device)
        inp = torch.cat([ids, mask], dim=1)
        logits = model(input_ids=inp,
                       attention_mask=torch.ones_like(inp)).logits
        nxt = logits[0, -1].argmax(-1).item()
        if nxt in tokenizer.all_special_ids:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def main():
    set_seed()
    texts, files = read_corpus()
    chunks = make_chunks(texts)
    train_t, val_t, test_t = split_texts(chunks)
    print(f"[Split] train={len(train_t)} val={len(val_t)} test={len(test_t)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = tokenize_dataset(train_t, tokenizer)
    val_ds   = tokenize_dataset(val_t, tokenizer)

    experiments = [
        ("e2_bs4_lr5e-5", 2, 4, 5e-5),
        ("e3_bs4_lr5e-5", 3, 4, 5e-5),
        ("e3_bs6_lr3e-5", 3, 6, 3e-5),
    ]
    results, best = [], None
    for name, ep, bs, lr in experiments:
        r = train_experiment(name, train_ds, val_ds, tokenizer, ep, bs, lr)
        results.append(r)
        if best is None or r["perplexity"] < best["perplexity"]:
            best = r

    print("\n[Таблица экспериментов]")
    print(pd.DataFrame(
        [{k: v for k, v in r.items() if k != "trainer"} for r in results]
    ).to_string(index=False))

    OUTPUT_DIR.mkdir(exist_ok=True)
    best["trainer"].save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"[Save] лучшая '{best['name']}' -> {OUTPUT_DIR}/")

    best_model = AutoModelForMaskedLM.from_pretrained(str(OUTPUT_DIR))
    acc = masked_accuracy(best_model, tokenizer, test_t)
    print(f"[Test] masked accuracy: {acc:.2%}")

    demo_prompts = [
        "def preprocess(self, text):",
        "self.morph = pymorphy2.",
        "df['aspect_graphics'] = ",
        "model = AutoModelForSequenceClassification.from_pretrained(",
        "return classification_report(y_test,",
    ]
    demo = []
    print("\n[Demo]")
    for prompt in demo_prompts:
        out = generate_completion(best_model, tokenizer, prompt, 10)
        demo.append({"prompt": prompt, "completion": out})
        print(f"  {prompt!r} -> {out!r}")

    report = {
        "project_files": [str(p) for p in files],
        "corpus_size": len(chunks),
        "split": {"train": len(train_t), "val": len(val_t), "test": len(test_t)},
        "experiments": [{k: v for k, v in r.items() if k != "trainer"}
                        for r in results],
        "best": {k: v for k, v in best.items() if k != "trainer"},
        "test_masked_accuracy": acc,
        "demo": demo,
    }
    RESULTS_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[Report] -> {RESULTS_JSON}")


if __name__ == "__main__":
    main()