from datasets import load_dataset, Dataset
from helpers.variables import SRC_DIR


def prepare_pubmed_dataset():
    dataset = load_dataset("PatronusAI/HaluBench", split="test").filter(
        lambda x: x["source_ds"] == "pubmedQA"
    )
    passages = dataset.select_columns(["passage"])

    if not isinstance(passages, Dataset):
        raise TypeError("Expected a Dataset object")

    for passage in passages:
        print(passage)

    passages.to_json(SRC_DIR + "input_data/pubmed.json", lines=False)
