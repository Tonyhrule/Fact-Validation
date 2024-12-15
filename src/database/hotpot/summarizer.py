import asyncio
from uuid import uuid4
from datasets import load_dataset, Dataset
from helpers.dbscan import cluster
from helpers.oai import async_gpt_calls, get_embeddings
from helpers.pc import upsert_index


def get_statement_prompt(context: str):
    return f"""Convert the following text into a series of concise, standalone statements.
This should be in bullet-points (- ).
Each statement should be a complete thought and should NOT reference any other bullet points or wider context.
This means that each statement should be able to stand alone and make sense.
For a statement to stand alone, it should contain all the context it needs to be understood.
Each statement MUST include its FULL setting (eg. In xyz, abc happened), and this setting must be specific (eg. Superbowl 50 instead of Superbowl).
ALWAYS use proper nouns if possible. (eg. "Superbowl 50" instead of "the game")

Text:
{context}"""


def get_compress_prompt(cluster: list[str]):
    statements = "\n".join(cluster)
    return f"""Combine the following statements into ONE statement.
The statement should be relatively short.

Statements:
{statements}"""


async def hotpot_summarize():
    dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="train")

    data = dataset.select_columns(["id", "context", "question", "supporting_facts"]).select(range(150))  # type: ignore

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
    for id, used_contexts, context in zip(
        data["id"], data["supporting_facts"], data["context"]
    ):
        for context_id in used_contexts["title"]:
            context_id_to_ids[context_id].append(id)

    print("Getting statements...")

    new_contexts = [
        str(x).replace("\n- ", "\n").strip().removeprefix("- ")
        for x in await async_gpt_calls(
            [
                get_statement_prompt(context)
                for context in context_id_to_raw_context.values()
            ],
            progress_bar=True,
        )
    ]

    statements = [
        {
            "statement": statement.strip(),
            "id": str(uuid4()),
            "questions": context_id_to_ids[context_id],
        }
        for statement_list, context_id in zip(
            new_contexts, context_id_to_raw_context.keys()
        )
        for statement in statement_list.split("\n")
        if statement.strip() != ""
    ]

    statement_id_to_statement = {statement["id"]: statement for statement in statements}

    print("Getting embeddings...")

    embeddings = await get_embeddings(
        [statement["statement"] for statement in statements], progress_bar=True
    )

    for statement, embedding in zip(statements, embeddings):
        statement["vector"] = embedding.vector

    print("Clustering embeddings...")

    clusters = cluster(
        [
            {"vector": embedding.vector, "id": statement["id"]}
            for embedding, statement in zip(embeddings, statements)
        ],
    )

    print("Compressing clusters...")

    no_compress = [cluster[0] for cluster in clusters if len(cluster) == 1]
    to_compress = [cluster for cluster in clusters if len(cluster) > 1]

    compressions = [
        str(x)
        for x in await async_gpt_calls(
            [
                get_compress_prompt(
                    [
                        statement_id_to_statement[statement_id]["statement"]
                        for statement_id in cluster
                    ]
                )
                for cluster in to_compress
            ],
            progress_bar=True,
        )
    ]

    print("Updating embeddings...")

    compression_embeddings = await get_embeddings(compressions, progress_bar=True)

    print("Upserting embeddings...")

    final_contexts = [
        {
            "id": statement_id,
            "values": statement_id_to_statement[statement_id]["vector"],
            "metadata": {
                "content": statement_id_to_statement[statement_id]["statement"],
                "ids": statement_id_to_statement[statement_id]["questions"],
            },
        }
        for statement_id in no_compress
    ]

    final_contexts += [
        {
            "id": str(uuid4()),
            "values": embedding.vector,
            "metadata": {
                "content": content,
                "ids": list(
                    set(
                        [
                            question
                            for statement_id in statement_ids
                            for question in statement_id_to_statement[statement_id][
                                "questions"
                            ]
                        ]
                    )
                ),
            },
        }
        for embedding, statement_ids, content in zip(
            compression_embeddings, to_compress, compressions
        )
    ]

    upsert_index("hotpot_summarized", final_contexts)

    print("Done")
