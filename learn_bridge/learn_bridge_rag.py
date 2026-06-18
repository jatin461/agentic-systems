import os
import json
import urllib.request
from typing import List, Dict, Any
import chromadb
# from google import genai

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


def main():
    print("=== LearnBridge Playbook RAG ===\n")

    # ----- OFFLINE: build the knowledge base (Steps A–E) -----
    download_policy_files()


if __name__ == "__main__":
    main()