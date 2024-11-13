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

    context_to_UUID = {}

    for context_obj in data["context"]:
        for context in context_obj["contexts"]:
            if context not in context_to_UUID:
                context_to_UUID[context] = str(uuid4())
                contexts.append(context)

    print("Getting embeddings...")

    embeddings = await get_embeddings(contexts)

    pubid_to_context_uuids = {}

    for item in list(data):
        pubid = str(item["pubid"])  # type: ignore
        context_uuids = [context_to_UUID[context] for context in item["context"]["contexts"]]  # type: ignore
        pubid_to_context_uuids[pubid] = context_uuids

    print("Upserting embeddings...")

    upsert_index(
        "pubmed_raw",
        [
            {
                "id": context_to_UUID[context],
                "values": embedding.vector,
                "metadata": {
                    "content": context,
                },
            }
            for embedding, context in zip(embeddings, contexts)
        ],
    )

    save_json("data/pubmed_raw_context_map.json", pubid_to_context_uuids)

    print("Done")
