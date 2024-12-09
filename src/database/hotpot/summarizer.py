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


async def hotpot_summarize():
    dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="train")

    data = dataset.select_columns(["id", "context", "question", "supporting_facts"]).select(range(15))  # type: ignore

    if not isinstance(data, Dataset):
        raise TypeError("Expected a Dataset object")

    context_id_to_ids: dict[str, list[str]] = {}
    context_id_to_raw_context: dict[str, str] = {}

    for id, used_contexts, context in zip(
        data["id"], data["supporting_facts"], data["context"]
    ):
        for context_id, context in zip(context["title"], context["sentences"]):
            if context_id not in context_id_to_raw_context:
                context_id_to_raw_context[context_id] = "".join(context)
                context_id_to_ids[context_id] = []
        for context_id in used_contexts["title"]:
            context_id_to_ids[context_id].append(id)

    new_contexts = [
        str(x)
        for x in await async_gpt_calls(
            [
                get_summarize_prompt(context)
                for context in context_id_to_raw_context.keys()
            ],
            progress_bar=True,
        )
    ]

    context_id_to_new_context = {
        context_id: new_context
        for context_id, new_context in zip(
            context_id_to_raw_context.keys(), new_contexts
        )
    }

    embeddings = await get_embeddings(list(context_id_to_raw_context.values()))

    context_id_to_embedding = {
        id: embedding
        for id, embedding in zip(context_id_to_raw_context.keys(), embeddings)
    }

    print("Clustering embeddings...")

    clusters = cluster(
        [
            {"vector": embedding.vector, "id": id}
            for id, embedding in context_id_to_embedding.items()
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
            "values": context_id_to_embedding[context_id].vector,
            "metadata": {
                "content": context_id_to_new_context[context_id],
                "ids": context_id_to_ids[context_id],
            },
        }
        for i, context_id in enumerate(no_compress)
    ]

    final_contexts += [
        {
            "id": str(len(no_compress) + i),
            "values": embedding.vector,
            "metadata": {
                "content": content,
                "ids": [
                    id
                    for context_id in contexts_ids
                    for id in context_id_to_ids[context_id]
                ],
            },
        }
        for i, embedding, contexts_ids, content in zip(
            range(len(to_compress)), compression_embeddings, to_compress, compressions
        )
    ]

    upsert_index("hotpot_summarized", final_contexts)

    print("Done")
