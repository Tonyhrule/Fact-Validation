import asyncio
from uuid import uuid4
from datasets import load_dataset, Dataset

from helpers.data import save_json
from helpers.oai import get_embeddings
from helpers.pc import upsert_index


async def pubmed_raw():
    dataset = load_dataset("qiaojin/PubMedQA", name="pqa_labeled", split="train")

    data = dataset.select_columns(["question", "context", "pubid"])

    if not isinstance(data, Dataset):
        raise TypeError("Expected a Dataset object")

    contexts = []

    for context_obj in data["context"]:
        contexts.append(" ".join(context_obj["contexts"]))

    print("Getting embeddings...")

    embeddings = await get_embeddings(contexts)

    print("Upserting embeddings...")

    upsert_index(
        "pubmed_raw",
        [
            {
                "id": str(data["pubid"][i]),
                "values": embedding.vector,
                "metadata": {
                    "content": context,
                },
            }
            for i, (embedding, context) in enumerate(zip(embeddings, contexts))
        ],
    )

    print("Done")
