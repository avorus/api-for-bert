## Шаги по установке
1. Убедитесь, что у вас установлен Python 3.7 или выше.
2. Установите необходимые библиотеки:
```bash
pip install fastapi uvicorn transformers torch
```

## Сохранение модели
Если у вас еще нет сохраненной модели и токенайзера, вы можете сохранить их следующим образом:
```python
from transformers import BertTokenizer, BertForSequenceClassification
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Сохранение модели и токенайзера
save_path = "./saved_model"
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
```
## Запуск
Запустите приложение командой:
```bash
uvicorn main:app --reload
```

Теперь API будет доступен по адресу http://127.0.0.1:8000/predict/, и вы сможете отправлять POST запросы с текстом для получения предсказаний от сохраненной модели BERT.

## Примечание
Документация FastAPI</br>
https://github.com/tiangolo/fastapi
