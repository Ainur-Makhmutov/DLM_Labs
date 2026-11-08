# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import sqlite3
import pandas as pd
import uvicorn
import os

app = FastAPI(title="Football NL2SQL API")

# --- НАСТРОЙКИ ПУТЕЙ ---
MODEL_DIR = "codet5_nl2sql"
DB_PATH = "project_code/database.sqlite"

# Загрузка обученной модели при старте API
print("Загрузка обученной модели...")
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Папка с моделью {MODEL_DIR} не найдена. Сначала запустите обучение!")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
model.eval()
print("Модель успешно загружена!")


# Схемы данных для валидации запросов и ответов
class QueryRequest(BaseModel):
    text: str


class QueryResponse(BaseModel):
    generated_sql: str
    execution_result: list | str
    error: str | None = None


@app.post("/generate", response_model=QueryResponse)
def generate_and_execute(request: QueryRequest):
    # 1. Генерация SQL-запроса
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256, num_beams=4)

    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 2. Безопасность: проверяем, что сгенерирован только SELECT
    # Это защищает БД от случайного удаления данных (DELETE/DROP)
    if not sql_query.strip().upper().startswith("SELECT"):
        return QueryResponse(
            generated_sql=sql_query,
            execution_result="Выполнение заблокировано.",
            error="Разрешены только SELECT-запросы."
        )

    # 3. Выполнение запроса в БД
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(sql_query, conn)
        conn.close()

        # Преобразуем DataFrame в список словарей для JSON
        result = df.to_dict(orient="records")

        # Ограничиваем вывод, чтобы API не зависло на огромных таблицах (например, Player_Attributes)
        if len(result) > 50:
            result = result[:50]
            result.append({"info": "Вывод обрезан до 50 строк для экономии трафика."})

        return QueryResponse(generated_sql=sql_query, execution_result=result)

    except Exception as e:
        # Если модель сгенерировала неверный SQL (например, опечатку в названии таблицы)
        return QueryResponse(
            generated_sql=sql_query,
            execution_result=[],
            error=f"Ошибка выполнения SQL: {str(e)}"
        )


if __name__ == "__main__":
    # Запуск сервера на localhost, порт 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)