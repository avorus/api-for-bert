from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = FastAPI()

# Путь к сохраненной модели и токенайзеру
model_path = "./saved_model"

# Загрузка модели и токенайзера
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

class TextInput(BaseModel):
    text: str

@app.post("/predict/")
async def predict(input: TextInput):
    try:
        # Токенизация входного текста
        inputs = tokenizer(input.text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")

        # Получение предсказаний модели
        with torch.no_grad():
            outputs = model(**inputs)

        # Извлечение вероятностей и меток классов
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        # Возврат результатов
        return {"text": input.text, "predicted_class": predictions.item(), "probabilities": probabilities.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Запуск приложения с помощью uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)