import asyncio
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from typing import Literal, Any

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

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)

class QueryResponse:
    def __init__(self, matches):
        self.matches = matches


def upsert_index(namespace: str, vectors: list[dict]):
    result = []

    chunks = chunk_list(vectors, 100)

    batches = chunk_list(chunks, 30)

    for batch in batches:
        requests = [
            index.upsert(namespace=namespace, vectors=chunk, async_req=True)
            for chunk in batch
        ]

        result += [request.result() for request in requests]  # type: ignore

    return result


async def embed_small(
    inputs: list[str],
    model="multilingual-e5-small",
    input_type: Literal["query", "passage"] = "passage",
    truncate=False,
):
    from helpers.oai import get_embeddings
    embeddings = await get_embeddings(inputs)
    return [embedding.vector for embedding in embeddings]


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


async def query_index(
    query: str,
    namespace: str,
    top_k=5,
    include_metadata=False,
    include_vector=False,
    min_score=0.0,
    filter={},
):
    embedding = await get_embedding(query)
    response = index.query(
        namespace=namespace,
        vector=embedding.vector,
        top_k=top_k,
        include_metadata=include_metadata,
        include_values=include_vector,
        filter=filter,
    )
    response_obj = QueryResponse(response.matches)
    return [match for match in response_obj.matches if match.score >= min_score]


async def async_query_index(
    query: list[float] | str,
    namespace: str,
    top_k=5,
    include_metadata=False,
    include_vector=False,
    min_score=0.0,
    progress: Progress | None = None,
    filter={},
):
    async def run(query, namespace, top_k, include_metadata, include_vector, filter):
        if isinstance(query, str):
            query = (await get_embedding(query)).vector
        response = index.query(
            namespace=namespace,
            vector=query,
            top_k=top_k,
            include_metadata=include_metadata,
            include_values=include_vector,
            filter=filter,
        )
        if progress:
            progress.increment()
        return [match for match in response.matches if match.score >= min_score]

    try:
        result = await asyncio.wait_for(
            run(query, namespace, top_k, include_metadata, include_vector, filter),
            timeout=60,
        )
        return result
    except asyncio.TimeoutError:
        return []
    except Exception as e:
        print("Error querying index:", e)
        return []


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

    p.finish() if p else None

    return [
        [match for match in response if match.score >= min_score]
        for response in responses
    ]


async def query_batches(
    queries: list[list[str]],
    namespace: str,
    top_k=10,
    include_metadata=False,
    include_vector=False,
    min_score=0.0,
    progress: bool = False,
):
    flattened_results = await multiple_queries(
        [query for batch in queries for query in batch],
        namespace,
        top_k,
        include_metadata,
        include_vector,
        min_score,
        progress,
    )

    unflattened_results = []
    i = 0

    for batch in queries:
        to_add = []
        for _ in batch:
            to_add += flattened_results[i]
            i += 1
        to_add.sort(key=lambda x: x.score, reverse=True)
        duplicates_removed = []
        for match in to_add:
            if match.id not in [m.id for m in duplicates_removed]:
                duplicates_removed.append(match)

        unflattened_results.append(duplicates_removed)

    return unflattened_results


def content_from_query_result(result: QueryResponse | list) -> list[str]:
    matches = result.matches if isinstance(result, QueryResponse) else result
    return [
        match.metadata["content"] 
        for match in matches if hasattr(match, 'metadata') and "content" in match.metadata
    ]


def get_namespace_size(namespace: str):
    stats = index.describe_index_stats()
    if namespace in stats.namespaces:
        return stats.namespaces[namespace].vector_count
    return 0
