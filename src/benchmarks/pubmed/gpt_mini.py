from datasets import load_dataset, Dataset
from helpers.oai import batch_gpt_call


def get_prompt(data):
    context = "\n\n".join(data["context"]["contexts"])

    return {
        "prompt": f"""Given the following text:
{context}

Answer the following question as a single (yes, no, maybe):
{data["question"]}"""
    }


def run_4o_batch():
    dataset = load_dataset("qiaojin/PubMedQA", name="pqa_labeled", split="train")

    data = dataset.select_columns(["question", "context"])

    if not isinstance(data, Dataset):
        raise TypeError("Expected a Dataset object")

    prompts = data.map(get_prompt)["prompt"]

    batch_gpt_call("PubMed GPT-4o-mini benchmark", prompts)

    print(
        "PubMed GPT-4o-mini benchmark processing. Wait for the batch to complete then run the next step."
    )
