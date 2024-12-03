import asyncio
from datasets import load_dataset, Dataset

from helpers.oai import get_embeddings
from helpers.pc import upsert_index


async def squad_raw():
    dataset = load_dataset("rajpurkar/squad", split="train")

    data = dataset.select_columns(["id", "context", "question", "answers"])

    if not isinstance(data, Dataset):
        raise TypeError("Expected a Dataset object")

    context_map: dict[str, list[str]] = {}
    contexts: list[str] = []

    for id, context in zip(data["id"], data["context"]):
        if context not in context_map:
            context_map[context] = []
            contexts.append(context)
        context_map[context].append(id)

    print("Getting embeddings...")

    embeddings = await get_embeddings(contexts)

    print("Upserting embeddings...")

    upsert_index(
        "squad_raw",
        [
            {
                "id": str(i),
                "values": embedding.vector,
                "metadata": {
                    "content": context,
                    "ids": context_map[context],
                },
            }
            for i, (embedding, context) in enumerate(zip(embeddings, contexts))
        ],
    )

    print("Done")
