from pydantic import BaseModel, validator
from peft import PeftModel, PeftConfig
from transformers import T5ForConditionalGeneration, AutoTokenizer
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)



peft_model_id = "deutsche-welle/t5_large_peft_wnc_debiaser"
config = PeftConfig.from_pretrained(peft_model_id)

model = T5ForConditionalGeneration.from_pretrained(config.base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model = PeftModel.from_pretrained(model, peft_model_id)
model.eval()


def prepare_input(sentence: str):
    input_ids = tokenizer(sentence, max_length=256, return_tensors="pt").input_ids
    return input_ids


def inference(sentence: str) -> str:
    input_data = prepare_input(sentence=sentence)
    input_data = input_data.to(model.device)
    outputs = model.generate(inputs=input_data, max_length=256)
    result = tokenizer.decode(token_ids=outputs[0], skip_special_tokens=True)
    return result

class Response(BaseModel):
    generated_text: str


@app.get("/debias", response_model=Response)
def predict_subjectivity(sentence: str):
    result = inference(f"debias: {sentence} </s>")
    return {"generated_text": result}