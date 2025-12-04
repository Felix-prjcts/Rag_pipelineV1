import glob
from pathlib import Path
import numpy as np
import unicodedata
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SearchParams
from transformers import pipeline


# 1. Charger et nettoyer les documents -----------------------------------------

def clean_text(raw: str) -> str:
    """Nettoyage léger : normalisation unicode + suppression ASCII + compression d'espaces."""
    normalized = unicodedata.normalize("NFKD", raw)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"\s+", " ", ascii_text)
    return cleaned.strip()


def load_documents(data_dir="data"):
    docs = []
    for path in glob.glob(f"{data_dir}/*.txt"):
        raw = Path(path).read_text(errors="ignore")
        text = clean_text(raw)
        docs.append({"id": Path(path).name, "text": text})
    return docs


# 2. Chunking (splitter anglais) -----------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def make_chunks(docs):
    chunks = []
    for doc in docs:
        for i, chunk in enumerate(splitter.split_text(doc["text"])):
            chunks.append({
                "doc_id": doc["id"],
                "chunk_id": f'{doc["id"]}_{i}',
                "text": chunk
            })
    return chunks


# 3. Embeddings + Index Qdrant -------------------------------------------------

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def build_qdrant_index(chunks, collection_name="demo_rag"):
    client = QdrantClient(":memory:")
    dim = embed_model.get_sentence_embedding_dimension()

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )

    texts = [c["text"] for c in chunks]
    vectors = embed_model.encode(texts)

    # DEBUG embeddings
    print("\n===== DEBUG EMBEDDINGS =====")
    print(f"Nombre de vecteurs : {len(vectors)}")
    print(f"Dimension d'un vecteur : {len(vectors[0])}")
    print("5 premières composantes du 1er vecteur :", vectors[0][:5])
    print("Norme L2 du 1er vecteur :", np.linalg.norm(vectors[0]))
    print("===== FIN DEBUG EMBEDDINGS =====\n")

    points = [
        PointStruct(
            id=i,
            vector=v.tolist(),
            payload={
                "doc_id": c["doc_id"],
                "chunk_id": c["chunk_id"],
                "text": c["text"]
            }
        )
        for i, (c, v) in enumerate(zip(chunks, vectors))
    ]

    client.upsert(collection_name=collection_name, points=points)
    return client, collection_name


# 4. Retrieval dense pur (sans keywords) 

def search_chunks(client, collection_name, query, k=5):
    q_vec = embed_model.encode([query])[0]

    # DEBUG vecteur de requête
    print("\n===== DEBUG REQUETE =====")
    print(f"Requête : {query}")
    print("Dimension :", len(q_vec))
    print("Premières composantes :", q_vec[:5])
    print("Norme L2 :", np.linalg.norm(q_vec))
    print("===== FIN DEBUG REQUETE =====\n")

    q_vec = q_vec.tolist()

    res = client.query_points(
        collection_name=collection_name,
        query=q_vec,
        limit=k,
        search_params=SearchParams(hnsw_ef=128),
        with_payload=True,
        with_vectors=False,
    )

    scored = []
    print("===== DEBUG RESULTATS RETRIEVAL =====")
    for r in res.points:
        print(f"Score = {r.score:.4f}")
        print(f"Chunk (doc {r.payload['doc_id']} | {r.payload['chunk_id']}) :")
        print(r.payload["text"][:200], "...")
        print("-" * 60)

        scored.append({
            "text": r.payload["text"],
            "doc_id": r.payload["doc_id"],
            "chunk_id": r.payload["chunk_id"],
            "score_raw": r.score,
            "score_total": r.score,  # DENSE PUR = score brut
        })

    print("===== FIN DEBUG RESULTATS RETRIEVAL =====\n")
    scored.sort(key=lambda x: x["score_total"], reverse=True)
    return scored


def build_context(chunks, top_n=3):
    parts = [c["text"] for c in chunks[:top_n]]
    return "\n\n---\n\n".join(parts)


# 5. LLM HuggingFace ------------------------------------------------------------

qa_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

def answer_query(query, client, collection_name, top_k=5):
    retrieved = search_chunks(client, collection_name, query, k=top_k)
    context = build_context(retrieved)

    prompt = (
        "You are an assistant that answers ONLY based on the context below.\n"
        "If the information is not in the context, say that you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    out = qa_model(prompt, max_length=256)[0]["generated_text"]
    return out, retrieved


# 6. main -----------------------------------------------------------------------

def main():
    print("Chargement des documents...")
    docs = load_documents("data")
    if not docs:
        print(" Aucun fichier .txt trouvé dans le dossier data/.")
        return

    print(f"{len(docs)} documents trouvés. Chunking...")
    chunks = make_chunks(docs)
    print(f"{len(chunks)} chunks générés.")

    # DEBUG chunks
    print("\n===== EXEMPLES DE CHUNKS =====")
    for c in chunks[:3]:
        print(f"[DOC {c['doc_id']}] [CHUNK {c['chunk_id']}]")
        print(c["text"][:300], "...")
        print("-" * 40)
    print("===== FIN EXEMPLES DE CHUNKS =====\n")

    print("Construction de l'index Qdrant...")
    client, collection_name = build_qdrant_index(chunks)
    print("Index prêt ")

    print("\nTu peux maintenant poser des questions. Taper 'exit' pour quitter.\n")

    while True:
        query = input(" Question : ")
        if query.strip().lower() in ("exit", "quit"):
            break

        answer, retrieved = answer_query(query, client, collection_name)
        print("\n💬 Réponse :")
        print(answer)
        print("\n📄 Chunks utilisés :")
        for r in retrieved[:3]:
            print(f"- [{r['doc_id']}] score={r['score_total']:.3f} -> {r['text'][:120]}...")
        print("\n" + "-"*60 + "\n")


if __name__ == "__main__":
    main()
