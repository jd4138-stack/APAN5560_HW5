# HW5/app/main.py
from fastapi import FastAPI
from pydantic import BaseModel

from HW5.models.gpt2_qa import generate_with_llm as llm_generate

app = FastAPI(title="HW5: Text Generation API")


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


# ----- Baseline endpoint (/generate_text) -----
@app.post("/generate_text")
def generate_text(request: TextGenerationRequest):
    """
    Very simple baseline text generator.
    In your final submission, you can replace this
    with whatever you used in Module 3.
    """
    text = (request.start_word + " ") * request.length
    return {"generated_text": text.strip()}


# ----- RNN endpoint (/generate_with_rnn) -----
@app.post("/generate_with_rnn")
def generate_with_rnn(request: TextGenerationRequest):
    """
    Placeholder for your RNN-based generator from Module 7.
    For now it just echoes a placeholder string.
    Later, you can import your RNN model and generate real text.
    """
    return {
        "generated_text": f"[RNN placeholder] start='{request.start_word}', length={request.length}"
    }


# ----- NEW: LLM endpoint (/generate_with_llm) -----
@app.post("/generate_with_llm")
def generate_with_llm(request: TextGenerationRequest):
    """
    Generate text using a (fine-tuned) GPT-2 model.
    """
    generated_text = llm_generate(
        prompt=request.start_word,
        max_new_tokens=request.length,
    )
    return {"generated_text": generated_text}
