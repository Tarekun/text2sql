import json
import numpy as np
from pgvector.sqlalchemy import Vector
import requests
from sqlalchemy import Column, Float, String
from sqlalchemy.orm import sessionmaker
from time import time
from src.cache.postgres import Base, SCHEMA_NAME
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
    response = requests.post(
        f"{api_base}/api/embeddings", json={"model": model, "prompt": description}
    )
    response.raise_for_status()
    embedding = response.json()["embedding"]
    embedding_vector = np.array(embedding).tolist()
    print(f"vector size {len(embedding_vector)}    {np.array(embedding).shape}")

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
