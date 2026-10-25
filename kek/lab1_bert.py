import os, re, ast, math, json, sqlite3, textwrap, random, time
import numpy as np
import pandas as pd
import torchs
import glob
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments,
)

DB_PATH = "erp.db"
MODEL_NAME = "microsoft/codebert-base"

def check_db():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM orders", conn)
    conn.close()
    print(f"[DB] Прочитано {len(df)} записей из 'orders'.")
    return df
#корпус кода проекта
def extract_pairs(source_code: str):
    pairs = []
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"[AST] Ошибка парсинга: {e}")
        return pairs
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node) or ""
            body = [b for b in node.body
                    if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))]
            if not body:
                continue
            body_src = "\n".join(
                textwrap.dedent(ast.get_source_segment(source_code, b) or "")
                for b in body
            ).strip()
            if doc and body_src:
                pairs.append({"input": doc.strip(), "target": body_src, "kind": "docstring->body"})
            sig = f"def {node.name}({', '.join(a.arg for a in node.args.args)}):"
            pairs.append({"input": sig, "target": body_src, "kind": "signature->body"})

        if isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node) or ""
            if doc:
                pairs.append({"input": f"class {node.name}:", "target": doc, "kind": "class->docstring"})
    return pairs

def extract_comment_blocks(source_code: str):
    blocks = re.findall(r"(#\s*={5,}.*?\n(?:#.*\n)*)", source_code)
    return [{"input": "section header", "target": b.strip(), "kind": "comment"} for b in blocks]

def load_project_corpus():
    folder = "project_code"
    source_files = [
        "demand_forecast.py",
        "lab_4_test.py",
        "lab_5_instant.py",
        "lab_5_expert.py",
        "2_DS.py",
        "1.py"
    ]
    full_paths = [os.path.join(folder, f) for f in source_files]

    existing_files = [f for f in full_paths if os.path.exists(f)]
    if not existing_files:
        raise FileNotFoundError(f"Не найдены файлы проекта в папке '{folder}'!")
    print(f"[Corpus] Найдено файлов: {len(existing_files)}")
    for f in existing_files:
        print(f"        - {f}")
    pairs = []
    total_chars = 0
    for file in existing_files:
        with open(file, encoding="utf-8", errors="ignore") as f:
            src = f.read()
            total_chars += len(src)
            file_pairs = extract_pairs(src) + extract_comment_blocks(src)
            lines = [l for l in src.splitlines() if l.strip()]
            for i in range(0, max(1, len(lines) - 8), 4):
                chunk = "\n".join(lines[i:i + 8])
                if len(chunk) > 40:
                    file_pairs.append({"input": "code chunk", "target": chunk, "kind": "chunk"})
            pairs.extend(file_pairs)
    print(f"[Corpus] Всего символов: {total_chars}")
    print(f"[Corpus] Примеров: {len(pairs)}")
    return pairs, ""

def clean_code(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
def build_texts(pairs):
    texts = []
    for p in pairs:
        texts.append(clean_code(p["target"]))
        if p["input"] and p["input"] not in ("section header", "code chunk"):
            texts.append(clean_code(p["input"]))
    seen, uniq = set(), []
    for t in texts:
        if t and t not in seen:
            seen.add(t);
            uniq.append(t)
    return uniq
def split_corpus(texts, seed=42):
    random.seed(seed);
    random.shuffle(texts)
    n = len(texts)
    return texts[:int(0.8 * n)], texts[int(0.8 * n):int(0.9 * n)], texts[int(0.9 * n):]

def load_model_and_tokenizer():
    print(f"[Model] Загружаю {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

def tokenize_dataset(texts, tokenizer, max_len=256):
    def _tok(ex):
        return tokenizer(ex["text"], truncation=True, max_length=max_len, padding="max_length")

    return Dataset.from_dict({"text": texts}).map(_tok, batched=True, remove_columns=["text"])

def run_experiment(name, train_ds, val_ds, tokenizer, model, epochs, batch_size, lr):
    print(f"\n[Exp:{name}] epochs={epochs}, bs={batch_size}, lr={lr}")
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    args = TrainingArguments(
        output_dir=f"./codebert_{name}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="epoch",
        report_to="none",
        seed=42,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collator)
    t0 = time.time();
    trainer.train();
    train_time = time.time() - t0
    eval_out = trainer.evaluate()
    ppl = math.exp(eval_out["eval_loss"])
    print(f"[Exp:{name}] eval_loss={eval_out['eval_loss']:.4f}  ppl={ppl:.2f}  time={train_time:.1f}s")
    return {"name": name, "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "eval_loss": eval_out["eval_loss"], "perplexity": ppl,
            "train_time_sec": round(train_time, 1), "trainer": trainer}

@torch.no_grad()
def masked_accuracy(model, tokenizer, texts, n_samples=20):
    model.eval();
    correct = total = 0
    for t in texts[:n_samples]:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=128)
        ids = enc["input_ids"]
        if ids.size(1) < 5: continue
        mask_pos = list(range(2, ids.size(1) - 1, 5))
        masked = ids.clone()
        for p in mask_pos: masked[0, p] = tokenizer.mask_token_id
        out = model(input_ids=masked, attention_mask=enc["attention_mask"]).logits
        pred = out.argmax(-1)
        for p in mask_pos:
            total += 1
            correct += int(pred[0, p].item() == ids[0, p].item())
    return correct / total if total else 0.0

def demo_completion(model, tokenizer, prompt):
    model.eval()
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    ids = enc["input_ids"];
    ids[0, -1] = tokenizer.mask_token_id
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=enc["attention_mask"]).logits
    return tokenizer.decode([out[0, -1].argmax().item()])

def main():
    print("ЛАБОРАТОРНАЯ 1: ДООБУЧЕНИЕ CodeBERT НА ПРОЕКТЕ")
    check_db()
    pairs, _ = load_project_corpus()
    texts = build_texts(pairs)
    train_t, val_t, test_t = split_corpus(texts)
    print(f"[Split] train={len(train_t)}, val={len(val_t)}, test={len(test_t)}")

    tokenizer, _ = load_model_and_tokenizer()
    train_ds = tokenize_dataset(train_t, tokenizer)
    val_ds = tokenize_dataset(val_t, tokenizer)

    experiments = [
        ("e2_bs4_lr5e5", 2, 4, 5e-5),
        ("e3_bs4_lr5e5", 3, 4, 5e-5),
        ("e3_bs6_lr3e5", 3, 6, 3e-5),
    ]
    results, best = [], None
    for name, ep, bs, lr in experiments:
        _, m = load_model_and_tokenizer()
        r = run_experiment(name, train_ds, val_ds, tokenizer, m, ep, bs, lr)
        results.append({k: v for k, v in r.items() if k != "trainer"})
        if best is None or r["perplexity"] < best["perplexity"]:
            best = r

    print("\n[Сводная таблица]")
    print(pd.DataFrame(results).to_string(index=False))

    best_dir = "codebert_best"
    best["trainer"].save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"[Save] Лучшая модель ({best['name']}) -> {best_dir}/")

    best_model = AutoModelForMaskedLM.from_pretrained(best_dir)
    acc = masked_accuracy(best_model, tokenizer, test_t)
    print(f"\n[Test] Masked-token accuracy: {acc:.2%}")

    print("\n[Demo] Автодополнение в стиле проекта:")
    for prompt in [
        "df_clean['OrderDate'] = pd.to_datetime(",
        "self.model = CatBoostRegressor(",
        "y = np.log1p(",
    ]:
        print(f"  '{prompt}' -> '{demo_completion(best_model, tokenizer, prompt)}'")

    with open("lab1_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "corpus_size": len(texts),
            "split": {"train": len(train_t), "val": len(val_t), "test": len(test_t)},
            "experiments": results,
            "best": {k: v for k, v in best.items() if k != "trainer"},
            "test_masked_accuracy": acc,
        }, f, ensure_ascii=False, indent=2)
    print("\n[Report] Метрики -> lab1_results.json")
    print("=" * 60)

if __name__ == "__main__":
    main()