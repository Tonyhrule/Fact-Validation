import asyncio
from datasets import load_dataset, Dataset
from helpers.data import save_json
from helpers.oai import get_embeddings
from helpers.pc import upsert_index


async def hotpot_raw():
    dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="train")

    data = dataset.select_columns(["id", "context", "question", "supporting_facts"]).select(range(150))  # type: ignore

    if not isinstance(data, Dataset):
        raise TypeError("Expected a Dataset object")

    context_to_ids: dict[str, list[str]] = {}
    contexts: dict[str, str] = {}

    for id, used_contexts, context in zip(
        data["id"], data["supporting_facts"], data["context"]
    ):
        for context_id, context in zip(context["title"], context["sentences"]):
            if context_id not in contexts:
                contexts[context_id] = "".join(context)
                context_to_ids[context_id] = []

    for id, used_contexts, context in zip(
        data["id"], data["supporting_facts"], data["context"]
    ):
        for context_id in used_contexts["title"]:
            context_to_ids[context_id].append(id)

    embeddings = await get_embeddings(list(contexts.values()), progress_bar=True)

    upsert_index(
        "hotpot_raw",
        [
            {
                "id": id.encode("ascii", "ignore").decode("ascii"),
                "values": embedding.vector,
                "metadata": {
                    "content": context,
                    "ids": context_to_ids[id],
                },
            }
            for embedding, (id, context) in zip(embeddings, contexts.items())
            if id.encode("ascii", "ignore").decode("ascii") != ""
        ],
    )

    print("Done")
