from pathlib import Path
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_DIR = Path("lab1_model")
MAX_LEN = 256

app = FastAPI(title="Lab 1 — BERT base completion API")

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForMaskedLM.from_pretrained(str(MODEL_DIR))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


class CompleteRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 8


class MaskRequest(BaseModel):
    text: str  # текст с одним [MASK]


@torch.no_grad()
def complete(prompt: str, max_new_tokens: int = 8) -> str:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=MAX_LEN - max_new_tokens)
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


@torch.no_grad()
def fill_mask(text: str, top_k: int = 5):
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=MAX_LEN)
    logits = model(**{k: v.to(device) for k, v in enc.items()}).logits[0]
    mask_positions = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero()
    if len(mask_positions) == 0:
        return []
    pos = mask_positions[0].item()
    probs = torch.softmax(logits[pos], dim=-1)
    top = torch.topk(probs, top_k)
    return [
        {"token": tokenizer.decode([i.item()]).strip(), "prob": float(p)}
        for p, i in zip(top.values, top.indices)
    ]


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/complete")
def api_complete(req: CompleteRequest):
    return {"prompt": req.prompt, "completion": complete(req.prompt, req.max_new_tokens)}


@app.post("/mask")
def api_mask(req: MaskRequest):
    return {"text": req.text, "candidates": fill_mask(req.text)}