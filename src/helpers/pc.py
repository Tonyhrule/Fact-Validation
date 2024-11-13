import asyncio
import os
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone, GRPCVector
from pinecone_plugins.inference import Inference
from typing import Literal

from helpers.data import chunk_list

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")

if not PINECONE_API_KEY:
    raise Exception("Pinecone API Key not found")

if not PINECONE_HOST:
    raise Exception("Pinecone Host not found")

pinecone = Pinecone(api_key=PINECONE_API_KEY)
inference: Inference = pinecone.inference  # type: ignore
index = pinecone.Index(host=PINECONE_HOST)


def upsert_index(namespace: str, vectors: list[dict]):
    requests = [
        index.upsert(namespace=namespace, vectors=batch, async_req=True)
        for batch in chunk_list(vectors, 300)
    ]

    return [request.result() for request in requests]  # type: ignore


def embed(
    inputs: list[str],
    model="multilingual-e5-large",
    input_type: Literal["query", "passage"] = "passage",
    truncate=False,
):
    return inference.embed(
        model=model,
        inputs=inputs,
        parameters={
            "input_type": input_type,
            "truncate": "END" if truncate else "NONE",
        },
    )
