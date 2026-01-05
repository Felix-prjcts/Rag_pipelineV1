import numpy as np
import unicodedata
import re
import fitz  # PyMuPDF
# ML libs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import pipeline
from rank_bm25 import BM25Okapi

class RAGEngine:
    def __init__(self):
        # On charge les modeles au demarrage pour pas perdre de temps apres
        # minilm est rapide sur CPU, suffisant pour la demo
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.qa_model = pipeline("text2text-generation", model="google/flan-t5-base")
        
        # Splitter un peu agressif pour avoir du contexte precis
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, 
            chunk_overlap=50, 
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def clean_text(self, raw: str) -> str:
        # virer les accents et caracteres bizarres
        normalized = unicodedata.normalize("NFKD", raw)
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        # cleanup des espaces multiples qui cassent les tokenizers
        return re.sub(r"\s+", " ", ascii_text).strip()

    def extract_text_from_pdf(self, file_stream) -> str:
        try:
            # gestion cas fichier local (path) vs upload streamlit (bytes)
            if isinstance(file_stream, str):
                doc = fitz.open(file_stream)
            else:
                doc = fitz.open(stream=file_stream.getvalue(), filetype="pdf")
            
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n"
            return full_text
        except Exception as e:
            return f"Error reading PDF: {e}"

    def process_documents(self, raw_docs):
        # transforme les fichiers bruts en petits morceaux (chunks)
        chunks = []
        for filename, text in raw_docs:
            cleaned = self.clean_text(text)
            
            if not cleaned: continue # skip si vide
                
            for i, chunk in enumerate(self.splitter.split_text(cleaned)):
                chunks.append({
                    "doc_id": filename,
                    "chunk_id": f'{filename}_{i}',
                    "text": chunk,
                    "index": len(chunks)
                })
        return chunks

    def build_indices(self, chunks):
        # eviter le crash si le PDF est vide ou illisible
        if not chunks:
            print("Warning: 0 chunks to index")
            return None, None

        # Qdrant
        client = QdrantClient(":memory:")
        dim = self.embed_model.get_sentence_embedding_dimension()
        
        client.recreate_collection(
            collection_name="rag_collection",
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        
        # vectorisation en batch
        texts = [c["text"] for c in chunks]
        vectors = self.embed_model.encode(texts)
        
        # construction des points pour qdrant
        points = [
            PointStruct(id=i, vector=v.tolist(), payload=chunks[i])
            for i, v in enumerate(vectors)
        ]
        client.upsert(collection_name="rag_collection", points=points)

        # Setup BM25 (keyword)
        # on tokenise simple pour l'instant
        tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        return client, bm25

    def search(self, query, client, bm25, chunks, k=60):
        # C'est ici que la magie RRF opère (Dense + Sparse)
        
        # A. Vector Search
        q_vec = self.embed_model.encode([query])[0].tolist()
        dense_res = client.query_points(
            collection_name="rag_collection", query=q_vec, limit=10, with_payload=True
        ).points

        # B. Keyword Search
        tk_query = self._tokenize(query)
        bm25_scores = bm25.get_scores(tk_query)
        sparse_indices = np.argsort(bm25_scores)[::-1][:10] # top 10

        # C. Fusion des scores (RRF)
        fused_scores = {}
        
        # Poids vecteurs
        for rank, point in enumerate(dense_res):
            c_id = point.payload['chunk_id']
            if c_id not in fused_scores:
                fused_scores[c_id] = {"score": 0.0, "payload": point.payload}
            fused_scores[c_id]["score"] += 1 / (k + rank + 1)

        # Poids mots-cles
        for rank, idx in enumerate(sparse_indices):
            chunk_data = chunks[idx]
            c_id = chunk_data['chunk_id']
            if c_id not in fused_scores:
                fused_scores[c_id] = {"score": 0.0, "payload": chunk_data}
            fused_scores[c_id]["score"] += 1 / (k + rank + 1)

        # Tri final
        final_results = list(fused_scores.values())
        final_results.sort(key=lambda x: x["score"], reverse=True)
        
        # on renvoie juste le top 3 pour pas noyer le LLM
        return final_results[:3]

    def generate_answer(self, query, context):
        # hard limit pour eviter l'erreur 512 tokens de flan-t5
        max_chars = 1600 
        safe_context = context[:max_chars]
        
        # astuce: mettre la question first dans le prompt
        # ca evite que le modele hallucine si le contexte est tronqué
        prompt = (
            f"Question: {query}\n\n"
            "Using the text below, answer the question above.\n"
            f"Context:\n{safe_context}\n\n"
            "Answer:"
        )
        
        return self.qa_model(prompt, max_length=256)[0]["generated_text"]

    def _tokenize(self, text):
        # simple regex, a ameliorer plus tard si besoin
        return re.findall(r'\w+', text.lower())