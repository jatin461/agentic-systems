import os
import json
import urllib.request
from typing import List, Dict, Any
import chromadb
from google import genai

# =============================================================================
# CONFIGURATION — paths, models, and S3 URLs
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_DIR = os.path.join(BASE_DIR, "policy_data")           # Downloaded JSON files
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_playbook_db")   # Vector DB on disk
COLLECTION_NAME = "learnbridge_playbook"

EMBEDDING_MODEL = "gemini-embedding-2"
GENERATION_MODEL = "gemini-2.5-flash"

POLICY_URLS = {
    "attendance_policy.json": (
        "https://s13n-curr-images-bucket.s3.ap-south-1.amazonaws.com/"
        "iitr-as-260113/module2/Masterclass/001/data/attendance_policy.json"
    ),
    "assignment_policy.json": (
        "https://s13n-curr-images-bucket.s3.ap-south-1.amazonaws.com/"
        "iitr-as-260113/module2/Masterclass/001/data/assignment_policy.json"
    ),
    "evaluation_policy.json": (
        "https://s13n-curr-images-bucket.s3.ap-south-1.amazonaws.com/"
        "iitr-as-260113/module2/Masterclass/001/data/evaluation_policy.json"
    ),
}

gemini_client = genai.Client(
    api_key="AIzaSyA5TX5ZpYvklwSfB1ZJ4zAhZzbIIu04dQA"
    # os.getenv("GEMINI_API_KEY")
)


# =============================================================================
# PHASE 1 — OFFLINE: INGESTION (Step A)
# Load knowledge sources from S3 into policy_data/
# =============================================================================
def download_policy_files():
    """Document loader: fetch all playbook JSON files from the course S3 bucket."""
    os.makedirs(POLICY_DIR, exist_ok=True)
    for filename, url in POLICY_URLS.items():
        local_path = os.path.join(POLICY_DIR, filename)
        urllib.request.urlretrieve(url, local_path)
        print(f"Downloaded: {local_path}")


# =============================================================================
# PHASE 1 — OFFLINE: CHUNKING (Step B)
# Each JSON "section" becomes one chunk with id, text, and metadata
# =============================================================================
def load_policy_chunks_from_json() -> List[Dict[str, Any]]:
    """Parse JSON policies into a list of chunks ready for embedding."""
    all_chunks = []
    for filename in os.listdir(POLICY_DIR):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(POLICY_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            policy = json.load(file)

        policy_type = policy.get("policy_type", "general")
        policy_title = policy.get("policy_title", filename)

        for section in policy.get("sections", []):
            all_chunks.append(
                {
                    "id": f"{policy_type}_{section.get('section_id', 'unknown')}",
                    "text": (
                        f"{section.get('heading', '')}. "
                        f"{section.get('content', '')}"
                    ).strip(),
                    "metadata": {
                        "policy_type": policy_type,
                        "policy_title": policy_title,
                        "section_id": section.get("section_id", ""),
                        "heading": section.get("heading", ""),
                        "source_file": filename,
                    },
                }
            )
    print(f"Loaded {len(all_chunks)} chunks from JSON policies.")
    return all_chunks


# =============================================================================
# PHASE 1 — OFFLINE: EMBEDDINGS (Step C)
# Turn text into vectors so Chroma can measure semantic similarity
# =============================================================================

def create_embeddings(texts: List[str]) -> List[List[float]]:
    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts
    )

    if not response.embeddings:
        raise ValueError("Gemini returned no embeddings for the input texts.")

    embeddings = [e.values for e in response.embeddings]
    if len(embeddings) == len(texts):
        return embeddings

    # Some Gemini embedding models may return a single embedding for the entire
    # batch when a multi-input request is treated as one combined content item.
    if len(embeddings) == 1 and len(texts) > 1:
        individual_embeddings = []
        for text in texts:
            single_response = gemini_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
            )
            if not single_response.embeddings:
                raise ValueError(
                    "Gemini returned no embedding for a single text chunk."
                )
            individual_embeddings.append(single_response.embeddings[0].values)
        return individual_embeddings

    raise ValueError(
        f"Expected {len(texts)} embeddings but got {len(embeddings)} from Gemini."
    )
# =============================================================================
# PHASE 1 — OFFLINE: VECTOR DATABASE (Step D)
# Chroma persists to disk — survives after you close VS Code
# =============================================================================
def setup_vector_database():
    """Create or reconnect to the local Chroma collection (cosine similarity)."""
    os.makedirs(CHROMA_PATH, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # Compare vectors by angle, not raw distance
    )
    return collection


# =============================================================================
# PHASE 1 — OFFLINE: INDEXING (Step E)
# Store chunk text + metadata + embedding vectors in Chroma (upsert = safe re-run)
# =============================================================================
def index_policy_chunks(collection, chunks: List[Dict[str, Any]]):
    """Embed all chunks and write them into the vector database (knowledge base build)."""
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Batch embed all chunk texts in one API call
    embeddings = create_embeddings(texts)

    # upsert: insert new chunks or update existing ones with the same id
    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    print(f"Indexed {len(chunks)} chunks into ChromaDB.")


# =============================================================================
# PHASE 2 — RUNTIME: RETRIEVER (Step F)
# Given a student question, find the most similar playbook chunks
# =============================================================================
def retrieve_chunks(
    collection,
    query: str,
    top_k: int = 3,
    policy_type_filter: str = None,
) -> List[Dict[str, Any]]:
    """
    RETRIEVER component of RAG.
    1) Embed the question  2) Similarity search in Chroma  3) Return top_k chunks
  Optional policy_type_filter narrows search to one policy (e.g. only 'evaluation').
    """
    # Convert student question to the same vector space as stored chunks
    query_embedding = create_embeddings([query])[0]

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,  # How many chunks to send to the LLM (tune for quality)
        "include": ["documents", "metadatas", "distances"],
    }

    # Metadata filter — e.g. search only inside evaluation_policy.json content
    if policy_type_filter:
        query_kwargs["where"] = {"policy_type": policy_type_filter}

    results = collection.query(**query_kwargs)

    # Package results into a simple list of dicts for the generator
    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({"text": doc, "metadata": meta, "distance": dist})
    return retrieved


# =============================================================================
# DEBUG HELPER (Step G) — Inspect what the retriever found before trusting the answer
# =============================================================================
def print_retrieved_chunks(query: str, chunks: List[Dict[str, Any]]):
    """Print retrieved chunks so you can verify retrieval quality (garbage in → garbage out)."""
    print("\n" + "=" * 72)
    print(f"Student question: {query}")
    print("=" * 72)
    for i, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {i}")
        print(f"  Policy type : {chunk['metadata'].get('policy_type')}")
        print(f"  Heading     : {chunk['metadata'].get('heading')}")
        print(f"  Distance    : {chunk['distance']:.4f}")  # Lower = closer match (cosine)
        print(f"  Text        : {chunk['text'][:200]}...")


# =============================================================================
# PHASE 2 — RUNTIME: PROMPT BUILDER (Step H)
# Inject retrieved playbook text into the LLM prompt (grounding)
# =============================================================================
def build_grounded_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Combine retrieved excerpts + strict rules.
    Rule 2 forces a fixed fallback when the playbook has no answer (e.g. hostel curfew).
    """
    context = ""
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        context += (
            f"\nPlaybook excerpt {i} | "
            f"Policy: {meta.get('policy_title')} | "
            f"Section: {meta.get('heading')}\n"
            f"{chunk['text']}\n"
        )

    prompt = f"""You are a helpful student support assistant for LearnBridge Academy.
Answer using ONLY the playbook excerpts below.
Rules:
1. Do not invent rules that are not in the excerpts.
2. If the answer is not in the excerpts, reply exactly:
   I don't have that information in the LearnBridge student playbook.
3. Use simple, friendly language suitable for students.
4. Mention numbers, penalties, and deadlines exactly as written in the excerpts.

Playbook excerpts:
{context}

Student question:
{query}

Final answer:"""
    return prompt


# =============================================================================
# PHASE 2 — RUNTIME: GENERATOR (Step I)
# LLM writes the answer using ONLY the retrieved context
# =============================================================================

def generate_grounded_answer(query: str, chunks: List[Dict[str, Any]]) -> str:
    prompt = build_grounded_prompt(query, chunks)

    response = gemini_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )

    return response.text
# =============================================================================
# BASELINE COMPARISON (Step J) — Same question WITHOUT retrieval (often hallucinates)
# =============================================================================
def generate_answer_without_retrieval(query: str) -> str:
    response = gemini_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=query
    )

    return response.text


# =============================================================================
# END-TO-END RAG (Step K) — Retrieve → inspect → generate
# =============================================================================
def answer_with_rag(
    collection,
    query: str,
    top_k: int = 3,
    policy_type_filter: str = None,
) -> str:
    """Full RAG path: retriever + generator wired together."""
    chunks = retrieve_chunks(
        collection, query, top_k=top_k, policy_type_filter=policy_type_filter
    )
    print_retrieved_chunks(query, chunks)
    return generate_grounded_answer(query, chunks)


# =============================================================================
# TUNING LAB (Step L) — See how top_k changes retrieval breadth and answer quality
# =============================================================================
def top_k_experiment(collection, query: str):
    """Try top_k = 1, 2, 3 on the same question to feel precision vs recall trade-off."""
    print("\n" + "#" * 72)
    print("TOP-K EXPERIMENT")
    print("#" * 72)
    for k in [1, 2, 3]:
        print(f"\n--- top_k = {k} ---")
        chunks = retrieve_chunks(collection, query, top_k=k)
        answer = generate_grounded_answer(query, chunks)
        print(answer)


# =============================================================================
# MAIN — Run offline build once, then demo runtime Q&A and improvements
# =============================================================================
def main():
    print("=== LearnBridge Playbook RAG ===\n")

    # ----- OFFLINE: build the knowledge base (Steps A–E) -----
    download_policy_files()
    chunks = load_policy_chunks_from_json()
    collection = setup_vector_database()
    index_policy_chunks(collection, chunks)

    # ----- RUNTIME: test questions students might ask -----
    test_questions = [
        "What is the minimum attendance required for the final evaluation?",
        "What is the penalty if I submit an assignment 1 day late?",
        "What percentage of my grade comes from assignments?",
        "What is the hostel curfew time?",  # Out of scope — should trigger fallback (Rule 2)
    ]

    print("\n\n=== GROUNDED RAG ANSWERS ===")
    for question in test_questions:
        print("\n" + "*" * 72)
        answer = answer_with_rag(collection, question, top_k=3)
        print("\nFinal answer:\n", answer)

    # ----- Compare grounded vs LLM-only on one question -----
    print("\n\n=== COMPARISON: WITH vs WITHOUT RETRIEVAL ===")
    compare_q = "Can I resubmit an assignment if I scored 35%?"
    print(f"\nQuestion: {compare_q}")
    rag_chunks = retrieve_chunks(collection, compare_q, top_k=2)
    print("\n--- With RAG (grounded) ---")
    print(generate_grounded_answer(compare_q, rag_chunks))
    print("\n--- Without RAG (LLM only) ---")
    print(generate_answer_without_retrieval(compare_q))

    # ----- Metadata filter demo: search only evaluation policy -----
    print("\n\n=== METADATA FILTER: evaluation policy only ===")
    eval_q = "What is the passing mark for the final proctored evaluation?"
    answer_filtered = answer_with_rag(
        collection, eval_q, top_k=2, policy_type_filter="evaluation"
    )
    print("\nFiltered answer:\n", answer_filtered)

    # ----- Top-k tuning experiment -----
    print("\n\n=== TOP-K EXPERIMENT ===")
    top_k_experiment(
        collection,
        "How many days of approved leave can I take per module?",
    )


if __name__ == "__main__":
    main()