from benchmarks.pubmed.gpt_mini import run_4o_batch
from helpers.input import function_from_list

programs = [("PubMed QA: 4o-mini Batch", run_4o_batch)]

if __name__ == "__main__":
    function_from_list("What benchmark would you like to run?", programs)
