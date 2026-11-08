#Лабораторная 2
import os, json, math, sqlite3, random, time
import pandas as pd, numpy as np, torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments,
)
import json, shutil, os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
DB        = "project_code/database.sqlite"
SEL_PAIRS = "project_code/nl_sql_pairs.jsonl"
DML_PAIRS = "project_code/nl_sql_pairs_dml.jsonl"
SEED      = 42
MODEL     = "./codet5-base"
OUT_DIR   = "codet5_nl2sql"
TOK_DIR = "codet5_tokenizer_fixed"

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def split(select_pairs, dml_pairs):
    random.seed(SEED)
    sel = select_pairs[:]
    random.shuffle(sel)
    n = len(sel)
    tr_sel = sel[:int(.8*n)]
    va     = sel[int(.8*n):int(.9*n)]
    te     = sel[int(.9*n):]
    tr = tr_sel + dml_pairs
    random.shuffle(tr)
    return tr, va, te

def normalize(s):
    return " ".join(s.strip().rstrip(";").split()).lower()

def exact_match(pred, gold):
    return int(normalize(pred) == normalize(gold))

def exec_accuracy(preds, golds, conn):
    ok = total = 0
    for p, g in zip(preds, golds):
        if not g.strip().upper().startswith("SELECT"):
            continue
        total += 1
        try:
            rp = pd.read_sql(p, conn).fillna(0)
            rg = pd.read_sql(g, conn).fillna(0)
            ok += int(rp.equals(rg))
        except Exception:
            pass
    return ok / total if total else 0.0

def run_codet5(train_pairs, val_pairs, epochs=3, bs=8, lr=3e-5):
    # Проверяем, что папка с моделью существует
    if not os.path.exists(MODEL):
        raise FileNotFoundError(f"Папка с моделью не найдена по пути: {MODEL}")

    # Копируем токенизатор для фикса
    if os.path.exists(TOK_DIR):
        shutil.rmtree(TOK_DIR)
    shutil.copytree(MODEL, TOK_DIR)

    stm_path = os.path.join(TOK_DIR, "special_tokens_map.json")
    if os.path.exists(stm_path):
        with open(stm_path, encoding="utf-8") as f:
            stm = json.load(f)
        if "additional_special_tokens" in stm:
            original = stm["additional_special_tokens"]
            if any(isinstance(t, dict) for t in original):
                stm["additional_special_tokens"] = [
                    t["content"] if isinstance(t, dict) else t
                    for t in original
                ]
                with open(stm_path, "w", encoding="utf-8") as f:
                    json.dump(stm, f, indent=2, ensure_ascii=False)
                print("[Tokenizer] special_tokens_map.json исправлен (dict -> str)")

    tok = AutoTokenizer.from_pretrained(TOK_DIR)

    assert tok.pad_token_id == 0, f"pad_token_id={tok.pad_token_id}, expected 0"
    extra0 = tok.convert_tokens_to_ids("<extra_id_0>")
    assert extra0 != tok.unk_token_id, "<extra_id_0> mapped to <unk>"
    print(f"[Tokenizer] OK, vocab={len(tok)}, pad={tok.pad_token_id}")

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

    def _tok(ex):
        enc = tok(ex["nl"], max_length=128, truncation=True)
        lbl = tok(ex["sql"], max_length=256, truncation=True)
        enc["labels"] = lbl["input_ids"]
        return enc

    def to_ds(ps):
        return Dataset.from_list([{"nl": p["nl"], "sql": p["sql"]} for p in ps]) \
            .map(_tok, remove_columns=["nl", "sql"])

    tr_ds, va_ds = to_ds(train_pairs), to_ds(val_pairs)
    coll = DataCollatorForSeq2Seq(tok, model=model)
    args = Seq2SeqTrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        learning_rate=lr,
        logging_steps=20,
        save_strategy="no",
        eval_strategy="epoch",
        predict_with_generate=True,
        report_to="none",
        seed=SEED,
    )
    tr = Seq2SeqTrainer(model=model, args=args,
                        train_dataset=tr_ds, eval_dataset=va_ds,
                        processing_class=tok, data_collator=coll)
    t0 = time.time(); tr.train(); dt = time.time() - t0
    ev = tr.evaluate()
    ppl = math.exp(ev["eval_loss"])
    tr.save_model(OUT_DIR); tok.save_pretrained(OUT_DIR)
    return {"name": "codet5_seq2seq", "epochs": epochs, "bs": bs, "lr": lr,
            "eval_loss": ev["eval_loss"], "ppl": ppl, "time_s": round(dt, 1)}, model, tok

def main():
    sel = load_jsonl(SEL_PAIRS)
    dml = load_jsonl(DML_PAIRS)
    print(f"SELECT: {len(sel)}, DML: {len(dml)}")
    if not sel:
        raise RuntimeError("Нет SELECT-пар. Запустите build_nl_sql_corpus.py")

    tr_p, va_p, te_p = split(sel, dml)
    print(f"train={len(tr_p)} val={len(va_p)} test={len(te_p)}")

    rB, mT5, tT5 = run_codet5(tr_p, va_p)
    print(f"[EXP] loss={rB['eval_loss']:.4f} ppl={rB['ppl']:.2f} time={rB['time_s']}s")

    # тест: генерация + EM + Execution Accuracy
    conn = sqlite3.connect(DB)
    preds, golds = [], []
    mT5.eval()
    for p in te_p:
        enc = tT5(p["nl"], return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            out = mT5.generate(**enc, max_length=256, num_beams=4)
        preds.append(tT5.decode(out[0], skip_special_tokens=True))
        golds.append(p["sql"])
    em  = np.mean([exact_match(p, g) for p, g in zip(preds, golds)])
    exa = exec_accuracy(preds, golds, conn)
    conn.close()
    print(f"[TEST] ExactMatch={em:.2%}  ExecutionAccuracy={exa:.2%}")

    demo = [
        "Показать названия всех стран",
        "Сколько всего записей в таблице Match",
        "Вывести имена всех игроков весом больше 80",
        "Найти топ-5 самых высоких игроков",
        "Показать все команды с названием содержащим 'United'",
        "Вывести средний рост всех игроков",
    ]
    print("\n[DEMO]")
    for q in demo:
        enc = tT5(q, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            out = mT5.generate(**enc, max_length=256, num_beams=4)
        print(f"  Q: {q}\n  SQL: {tT5.decode(out[0], skip_special_tokens=True)}\n")

    report = {
        "corpus_size": len(sel) + len(dml),
        "select_size": len(sel),
        "dml_size": len(dml),
        "split": {"train": len(tr_p), "val": len(va_p), "test": len(te_p)},
        "experiments": [rB],
        "test": {"exact_match": float(em), "execution_accuracy": float(exa)},
    }
    json.dump(report, open("lab2_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n[OK] lab2_results.json записан")

if __name__ == "__main__":
    main()