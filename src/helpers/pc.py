import asyncio
import os
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone_plugins.inference import Inference
from pinecone.core.openapi.data.model.query_response import QueryResponse
from typing import Literal

from helpers.data import chunk_list
from helpers.oai import get_embedding, get_embeddings
from helpers.progress import Progress

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
        for batch in chunk_list(vectors, 100)
    ]

    return [request.result() for request in requests]  # type: ignore


async def embed_small(
    inputs: list[str],
    model="multilingual-e5-small",
    input_type: Literal["query", "passage"] = "passage",
    truncate=False,
):
    return [
        embedding["values"]
        for embedding in inference.embed(
            model=model,
            inputs=inputs,
            parameters={
                "input_type": input_type,
                "truncate": "END" if truncate else "NONE",
            },
        ).data
    ]


async def embed(
    inputs: list[str],
    model="multilingual-e5-large",
    input_type: Literal["query", "passage"] = "passage",
    truncate=False,
):
    responses = await asyncio.gather(
        *[
            embed_small(
                inputs=chunk,
                model=model,
                input_type=input_type,
                truncate=truncate,
            )
            for chunk in chunk_list(inputs, 96)
        ]
    )

    result: list[list[float]] = []

    for response in responses:
        result += response

    return result


def query_index(
    query: str,
    namespace: str,
    top_k=5,
    include_metadata=False,
    include_vector=False,
    min_score=0.0,
):
    embedding = get_embedding(query)
    response = index.query(
        namespace=namespace,
        vector=embedding.vector,
        top_k=top_k,
        include_metadata=include_metadata,
        include_values=include_vector,
    )
    response.matches = [match for match in response.matches if match.score >= min_score]
    return response


async def async_query_index(
    query: list[float],
    namespace: str,
    top_k=5,
    include_metadata=False,
    include_vector=False,
    min_score=0.0,
    progress: Progress | None = None,
):
    response = index.query(
        namespace=namespace,
        vector=query,
        top_k=top_k,
        include_metadata=include_metadata,
        include_vector=include_vector,
    )
    response.matches = [match for match in response.matches if match.score >= min_score]
    if progress:
        progress.increment()
    return response


async def multiple_queries(
    queries: list[str],
    namespace: str,
    top_k=10,
    include_metadata=False,
    include_vector=False,
    min_score=0.0,
    progress: bool = False,
):
    embeddings = await get_embeddings(queries)

    p = Progress(len(embeddings)) if progress else None

    responses = await asyncio.gather(
        *[
            async_query_index(
                embedding.vector,
                namespace,
                top_k=top_k,
                include_metadata=include_metadata,
                include_vector=include_vector,
                progress=p,
            )
            for embedding in embeddings
        ]
    )
    for response in responses:
        response.matches = [
            match for match in response.matches if match.score >= min_score
        ]

    p.finish() if p else None

    return responses


def content_from_query_result(result: QueryResponse) -> list[str]:
    return [match.metadata["content"] for match in result.matches]
