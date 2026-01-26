import json
import numpy as np
from pgvector.sqlalchemy import Vector
import requests
from sqlalchemy import Column, Float, String, select
from sqlalchemy.orm import sessionmaker, Session
from time import time
from src.cache.postgres import Base, SCHEMA_NAME, get_engine
from src.logger import logger


class QueryEmbeddings(Base):
    __tablename__ = "query_embeddings"
    __table_args__ = {"schema": SCHEMA_NAME}

    name = Column(String, primary_key=True)
    description = Column(String)
    query = Column(String)
    embedding = Column(Vector(768))
    # embedding = Column(Vector(4096))
    created_at = Column(Float, default=lambda: time())


def _embed_query(
    name: str,
    description: str,
    query: str,
    model: str,
    api_base: str,
) -> QueryEmbeddings:
    text = f"Query name: {name}\nDescription: {description}\n\nCode:\n{query}"
    embedding_vector = embed_text(text, model, api_base)

    return QueryEmbeddings(
        name=name,
        description=description,
        query=query,
        embedding=embedding_vector,
    )


def _persist_embedded_queries(queries: list[QueryEmbeddings], engine):
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        session.bulk_save_objects(queries)
        session.commit()
        logger.info(f"Successfully cached {len(queries)} queries")
    except Exception as e:
        session.rollback()
        logger.error(f"Error caching queries: {e}")
        raise e
    finally:
        session.close()


def cache_queries(filepath: str, engine):
    with open(filepath, "r") as f:
        content = f.read()
    queries_data: list[dict] = json.loads(content)
    result: list[QueryEmbeddings] = []

    for query in queries_data:
        embedding = _embed_query(
            name=query["name"],
            description=query["description"],
            query=query["query"],
            model="hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:F32",
            # model="qwen3-embedding:8b",
            api_base="http://192.168.178.82:11434",
        )
        result.append(embedding)

    _persist_embedded_queries(result, engine)


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


def top_k_lookup(user_question: str, k: int) -> list[QueryEmbeddings]:
    query_embedding = embed_text(
        user_question,
        "hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:F32",
        "http://192.168.178.82:11434",
    )

    with Session(get_engine()) as session:
        stmt = (
            select(QueryEmbeddings)
            .order_by(QueryEmbeddings.embedding.cosine_distance(query_embedding))
            .limit(k)
        )
        results = session.execute(stmt).scalars().all()

    return list(results)
