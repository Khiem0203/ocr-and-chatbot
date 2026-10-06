import logging
import os

from dotenv import load_dotenv
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

load_dotenv()

logger = logging.getLogger("milvus_store")

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "document_chunks")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

_connected = False


def _connect() -> None:
    global _connected
    if _connected:
        return
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    _connected = True


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def ensure_collection(name: str = None) -> Collection:
    name = name or COLLECTION_NAME
    _connect()
    if utility.has_collection(name):
        collection = Collection(name)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="bucket", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="object_key", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="page_index", dtype=DataType.INT64),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        ]
        schema = CollectionSchema(fields, description="OCR document chunks")
        collection = Collection(name, schema)
        collection.create_index(
            field_name="embedding",
            index_params={"index_type": "HNSW", "metric_type": "IP", "params": {"M": 16, "efConstruction": 200}},
        )
    collection.load()
    return collection


def delete_document_chunks(bucket: str, object_key: str, collection_name: str = None) -> None:
    collection = ensure_collection(collection_name)
    expr = f'bucket == "{_escape(bucket)}" && object_key == "{_escape(object_key)}"'
    collection.delete(expr)
    collection.flush()


def upsert_chunks(bucket: str, object_key: str, chunks: list, embeddings: list, collection_name: str = None) -> None:
    collection = ensure_collection(collection_name)
    delete_document_chunks(bucket, object_key, collection_name)

    data = [
        [bucket] * len(chunks),
        [object_key] * len(chunks),
        [c["chunk_index"] for c in chunks],
        [c["page_index"] for c in chunks],
        [c["text"][:8192] for c in chunks],
        embeddings,
    ]
    collection.insert(data)
    collection.flush()


def get_document_chunks(bucket: str, object_key: str, max_pages: int = None, collection_name: str = None) -> tuple:
    collection = ensure_collection(collection_name)
    base = f'bucket == "{_escape(bucket)}" && object_key == "{_escape(object_key)}"'
    fields = ["chunk_index", "page_index", "chunk_text"]

    if max_pages is None:
        rows = collection.query(expr=base, output_fields=fields, limit=16384)
        truncated = False
    else:
        rows = collection.query(expr=f"{base} && page_index < {int(max_pages)}", output_fields=fields, limit=16384)
        rest = collection.query(expr=f"{base} && page_index >= {int(max_pages)}", output_fields=["chunk_index"], limit=1)
        truncated = bool(rest)

    rows.sort(key=lambda r: r["chunk_index"])
    return rows, truncated


def search(query_embedding: list, top_k: int = 20, collection_name: str = None) -> list:
    collection = ensure_collection(collection_name)
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"ef": 64}},
        limit=top_k,
        output_fields=["bucket", "object_key", "chunk_index", "page_index", "chunk_text"],
    )
    hits = []
    for hit in results[0]:
        hits.append({
            "score": hit.score,
            "bucket": hit.entity.get("bucket"),
            "object_key": hit.entity.get("object_key"),
            "chunk_index": hit.entity.get("chunk_index"),
            "page_index": hit.entity.get("page_index"),
            "chunk_text": hit.entity.get("chunk_text"),
        })
    return hits
