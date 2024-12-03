import asyncio
from datasets import load_dataset, Dataset

from helpers import progress
from helpers.dbscan import cluster
from helpers.oai import async_gpt_calls, get_embeddings
from helpers.pc import upsert_index

NEWLINES = "\n\n"


def get_summarize_prompt(context: str):
    return f"""Give an EXTREMELY short summary of the text. Each sentence should have purpose. Make sure you're shortening the text instead of making it longer.

Text:
{context}"""


def get_compress_prompt(cluster: list[str]):
    return f"""Combine the following summaries into an EXTREMELY short summary. Each sentence should have purpose. Make sure you're shortening the text instead of making it longer.

Paragraphs:
{NEWLINES.join(cluster)}"""


async def squad_summarize():
    dataset = load_dataset("rajpurkar/squad", split="train")

    data = dataset.select_columns(["id", "context", "question", "answers"])

    if not isinstance(data, Dataset):
        raise TypeError("Expected a Dataset object")

    context_to_ids: dict[str, list[str]] = {}
    raw_contexts: list[str] = []

    for id, context in zip(data["id"], data["context"]):
        if context not in context_to_ids:
            context_to_ids[context] = []
            raw_contexts.append(context)
        context_to_ids[context].append(id)

    print("Getting summaries...")

    new_contexts = [
        str(x)
        for x in await async_gpt_calls(
            [get_summarize_prompt(context) for context in raw_contexts],
            progress_bar=True,
        )
    ]

    new_context_to_ids = {
        new_context: context_to_ids[raw_context]
        for new_context, raw_context in zip(new_contexts, raw_contexts)
    }

    print("Getting embeddings...")

    embeddings = await get_embeddings(new_contexts)

    context_to_embedding = {
        context: embedding for context, embedding in zip(new_contexts, embeddings)
    }

    print("Clustering embeddings...")

    clusters = cluster(
        [
            {"vector": embedding.vector, "id": context}
            for embedding, context in zip(embeddings, new_contexts)
        ]
    )

    print("Compressing clusters...")

    no_compress = [cluster[0] for cluster in clusters if len(cluster) == 1]
    to_compress = [cluster for cluster in clusters if len(cluster) > 1]

    compressions = [
        str(x)
        for x in await async_gpt_calls(
            [get_compress_prompt(cluster) for cluster in to_compress], progress_bar=True
        )
    ]

    print("Updating embeddings...")

    compression_embeddings = await get_embeddings(compressions)

    print("Upserting embeddings...")

    final_contexts = [
        {
            "id": str(i),
            "values": context_to_embedding[context].vector,
            "metadata": {
                "content": context,
                "ids": new_context_to_ids[context],
            },
        }
        for i, context in enumerate(no_compress)
    ]

    final_contexts += [
        {
            "id": str(len(no_compress) + i),
            "values": embedding.vector,
            "metadata": {
                "content": content,
                "ids": [
                    id for context in contexts for id in new_context_to_ids[context]
                ],
            },
        }
        for i, embedding, contexts, content in zip(
            range(len(to_compress)), compression_embeddings, to_compress, compressions
        )
    ]

    upsert_index("squad_summarized", final_contexts)

    print("Done")
