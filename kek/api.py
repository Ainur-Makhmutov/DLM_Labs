#uvicorn api:app --reload
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

MODEL_DIR = "./codebert_best"

app = FastAPI(title="Code Completion API (project style)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR)
model.eval()

class Prompt(BaseModel):
    code: str

@app.post("/complete")
def complete(p: Prompt):
    current_text = p.code
    generated_text = ""
    # Генерируем до 3 токенов
    for _ in range(3):
        enc = tokenizer(current_text, return_tensors="pt", truncation=True, max_length=128)
        ids = enc["input_ids"]
        # Добавляем [MASK] в конец, чтобы модель предсказала следующее слово
        mask_id = tokenizer.mask_token_id
        ids = torch.cat([ids, torch.tensor([[mask_id]])], dim=1)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        pred_token_id = out[0, -1].argmax().item()
        pred_token_str = tokenizer.decode([pred_token_id])
        if pred_token_id in tokenizer.all_special_ids and pred_token_id != tokenizer.mask_token_id:
            break
        if "#" in pred_token_str or "`" in pred_token_str:
            break
        generated_text += pred_token_str
        current_text += pred_token_str
        # Если сгенерировали закрывающую скобку или перенос строки - стоп
        if ")" in pred_token_str or "\n" in pred_token_str:
            break
    return {"prediction": generated_text}