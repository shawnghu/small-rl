"""FastAPI server hosting a HuggingFace classification model for reward scoring.

Serves any AutoModelForSequenceClassification model via a simple HTTP API.
Designed to be used with api_reward() from api_rewards.py.

Usage:
    REWARD_MODEL=distilbert/distilbert-base-uncased-finetuned-sst-2-english \
      uvicorn reward_server:app --host 0.0.0.0 --port 8100

Environment variables:
    REWARD_MODEL: HuggingFace model name (default: distilbert SST-2)
    REWARD_DEVICE: Device to run on (default: cuda)
    REWARD_MAX_LENGTH: Max tokenization length (default: 512)
"""

import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = os.environ.get(
    "REWARD_MODEL", "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
DEVICE = os.environ.get("REWARD_DEVICE", "cuda")
MAX_LENGTH = int(os.environ.get("REWARD_MAX_LENGTH", "512"))

model = None
tokenizer = None
id2label = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, id2label
    print(f"Loading model: {MODEL_NAME} on {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    id2label = model.config.id2label
    model.to(DEVICE)
    if DEVICE == "cuda":
        model.half()
    model.eval()
    model.requires_grad_(False)
    print(f"Model loaded. Labels: {id2label}")
    yield
    del model, tokenizer


app = FastAPI(lifespan=lifespan)


class ScoreRequest(BaseModel):
    texts: list[str]


class ScoreResult(BaseModel):
    label: str
    scores: dict[str, float]


class ScoreResponse(BaseModel):
    results: list[ScoreResult]
    model: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest):
    assert model is not None, "Model not loaded"
    assert tokenizer is not None, "Tokenizer not loaded"

    inputs = tokenizer(
        request.texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)

    results = []
    for i in range(len(request.texts)):
        scores = {id2label[j]: probs[i, j].item() for j in range(probs.shape[1])}
        top_label = max(scores, key=scores.get)
        results.append(ScoreResult(label=top_label, scores=scores))

    return ScoreResponse(results=results, model=MODEL_NAME)
