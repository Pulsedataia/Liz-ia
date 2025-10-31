import os
import json
import math
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Config
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

# Estado global simples
STATE = {
    "model": None,
    "emb": None,      # np.ndarray [N, D]
    "meta": None      # List[Dict]
}

class ChatRequest(BaseModel):
    message: str = Field(..., description="Pergunta do usuário sobre finanças/negócios")

class ChatResponse(BaseModel):
    ok: bool
    reply: str
    context: Optional[List[Dict[str, Any]]] = None

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: [1, D], b: [N, D]
    # ambos já normalizados
    return (a @ b.T).squeeze(0)  # [N]

def load_index():
    emb_path = DATA_DIR / "embeddings.npy"
    meta_path = DATA_DIR / "meta.json"
    if not emb_path.exists() or not meta_path.exists():
        raise RuntimeError("Índice não encontrado. Rode primeiro: python scripts/build_kb.py")

    emb = np.load(emb_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if emb.shape[0] != len(meta):
        raise RuntimeError("Tamanho de embeddings != tamanho do meta.")

    STATE["emb"] = emb
    STATE["meta"] = meta

def ensure_model():
    if STATE["model"] is None:
        STATE["model"] = SentenceTransformer(MODEL_NAME)

def retrieve(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    ensure_model()
    if STATE["emb"] is None or STATE["meta"] is None:
        load_index()

    q_emb = STATE["model"].encode([query], normalize_embeddings=True).astype(np.float32)  # [1, D]
    scores = cosine_sim(q_emb, STATE["emb"])  # [N]
    idx = np.argsort(-scores)[:top_k]

    hits: List[Dict[str, Any]] = []
    for i in idx:
        item = STATE["meta"][int(i)]
        hits.append({
            "score": float(scores[int(i)]),
            "id": item.get("id"),
            "pergunta": item.get("pergunta"),
            "resposta": item.get("resposta"),
            "tags": item.get("tags", [])
        })
    return hits

def craft_answer(question: str, hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return ("Ainda não tenho esse conceito na base. "
                "Você pode reformular a pergunta ou me dizer qual tópico deseja entender "
                "(ex.: SELIC, IPCA, câmbio, capital de giro, fluxo de caixa).")

    best = hits[0]
    answer = best["resposta"].strip()

    # Se houver material complementar, adiciona de forma sucinta
    if len(hits) > 1:
        extras = []
        for h in hits[1:]:
            extras.append(f"- {h['pergunta']}")
        if extras:
            answer += "\n\nVocê também pode querer saber:\n" + "\n".join(extras)

    return answer

app = FastAPI(title="Consultora Financeira Empresarial - Fase 1", version="0.1.0")

@app.on_event("startup")
def on_startup():
    load_index()  # pré-carrega índice para respostas rápidas

@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL_NAME,
        "items": 0 if STATE["meta"] is None else len(STATE["meta"])
    }

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    q = req.message.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Mensagem vazia.")
    hits = retrieve(q, top_k=TOP_K)
    reply = craft_answer(q, hits)
    ctx = [{"pergunta": h["pergunta"], "score": round(h["score"], 4)} for h in hits]
    return ChatResponse(ok=True, reply=reply, context=ctx)