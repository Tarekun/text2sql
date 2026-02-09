import requests
import numpy as np


def embed_text(
    text: str,
    model: str,
    api_base: str,
) -> list:
    response = requests.post(
        f"{api_base}/api/embeddings", json={"model": model, "prompt": text}
    )
    response.raise_for_status()
    embedding = response.json()["embedding"]
    embedding_vector = np.array(embedding)
    normalized = embedding_vector / np.linalg.norm(embedding_vector)
    return normalized.tolist()
