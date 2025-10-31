import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    kb_path = data_dir / "knowledge.json"

    if not kb_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {kb_path}")

    with kb_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    corpus = []
    meta = []
    for it in items:
        texto = (
            f"PERGUNTA: {it.get('pergunta','')}\n"
            f"RESPOSTA: {it.get('resposta','')}\n"
            f"TAGS: {', '.join(it.get('tags', []))}"
        )
        corpus.append(texto)
        meta.append({
            "id": it.get("id"),
            "pergunta": it.get("pergunta"),
            "resposta": it.get("resposta"),
            "tags": it.get("tags", [])
        })

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(corpus, normalize_embeddings=True).astype(np.float32)

    np.save(data_dir / "embeddings.npy", emb)
    (data_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Embeddings gerados: {emb.shape}, itens: {len(meta)}")
    print("Arquivos salvos em data/embeddings.npy e data/meta.json")

if __name__ == "__main__":
    main()
