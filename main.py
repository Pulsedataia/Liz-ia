from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("consultora-financeira")

# Constantes
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

# Estado global
STATE: Dict[str, Any] = {
    "model": None,
    "emb": None,
    "meta": None
}

# Modelos Pydantic
class ChatRequest(BaseModel):
    message: str = Field(..., description="Pergunta do usuário sobre finanças/negócios")

class ChatResponse(BaseModel):
    ok: bool
    reply: str
    context: Optional[List[Dict[str, Any]]] = None

# Funções utilitárias
def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a @ b.T).squeeze(0)

def load_index():
    emb_path = DATA_DIR / "embeddings.npy"
    meta_path = DATA_DIR / "meta.json"
    if not emb_path.exists() or not meta_path.exists():
        raise RuntimeError("Índice não encontrado. Rode: python scripts/build_kb.py")

    emb = np.load(emb_path)
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    if emb.shape[0] != len(meta):
        raise RuntimeError("Tamanho de embeddings != tamanho do meta.")

    STATE["emb"] = emb
    STATE["meta"] = meta
    logger.info("Índice carregado: %s embeddings, %s metas", emb.shape[0], len(meta))

def ensure_model():
    if STATE["model"] is None:
        logger.info("Carregando modelo de embeddings: %s", MODEL_NAME)
        STATE["model"] = SentenceTransformer(MODEL_NAME)

def retrieve(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    ensure_model()
    if STATE["emb"] is None or STATE["meta"] is None:
        load_index()

    q_emb = STATE["model"].encode([query], normalize_embeddings=True).astype(np.float32)
    scores = cosine_sim(q_emb, STATE["emb"])
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
                "Você pode reformular a pergunta ou indicar o tópico (SELIC, IPCA, câmbio, capital de giro, fluxo de caixa).")

    best = hits[0]
    answer = best["resposta"].strip()

    if len(hits) > 1:
        extras = [f"- {h['pergunta']}" for h in hits[1:]]
        if extras:
            answer += "\n\nVocê também pode querer saber:\n" + "\n".join(extras)

    return answer

# FastAPI
app = FastAPI(title="Consultora Financeira Empresarial - Fase 1", version="0.1.0")

# CORS — ajuste os domínios conforme onde seu front roda
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "http://localhost",
        "https://www.seu-dominio.com.br"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    # Tenta carregar o índice no startup, mas não derruba a API se não existir ainda
    try:
        load_index()
    except Exception as e:
        logger.warning("Startup sem índice pronto: %s", e)

@app.get("/health")
def health():
    items = 0 if STATE["meta"] is None else len(STATE["meta"])
    loaded = STATE["emb"] is not None and STATE["meta"] is not None
    return {
        "ok": True,
        "model": MODEL_NAME,
        "items": items,
        "index_loaded": loaded
    }

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    q = req.message.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Mensagem vazia.")
    try:
        hits = retrieve(q, top_k=TOP_K)
    except Exception as e:
        # Mensagem amigável quando o índice não está pronto
        raise HTTPException(
            status_code=503,
            detail=f"Base não carregada. Gere e carregue o índice: python scripts/build_kb.py. Detalhe: {e}"
        )
    reply = craft_answer(q, hits)
    ctx = [{"pergunta": h["pergunta"], "score": round(h["score"], 4)} for h in hits]
    return ChatResponse(ok=True, reply=reply, context=ctx)