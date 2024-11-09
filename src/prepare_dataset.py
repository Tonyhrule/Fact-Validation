from dataset_preparation.pubmed import prepare_pubmed_dataset
from helpers.input import function_from_list

datasets = [("Pubmed", prepare_pubmed_dataset)]

if __name__ == "__main__":
    function_from_list("Which dataset would you like to prepare?", datasets)
