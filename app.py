import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./en_ru_model"

print(f"Loading model from {model_path} ...")
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path).to(device)

app = FastAPI(title="EN→RU Translation Service")

class RequestText(BaseModel):
    text: str

@app.post("/translate")
def translate(item: RequestText):
    model.eval()
    with torch.no_grad():
        tokens = tokenizer(
            item.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)

        output = model.generate(
            **tokens,
            num_beams=5,
            max_length=256,
            early_stopping=True
        )

    translation = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"translation": translation}

if __name__ == "__main__":
    print("Starting FastAPI server")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
