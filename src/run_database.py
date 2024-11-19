from database.pubmed.raw import pubmed_raw
from database.pubmed.summarizer import pubmed_summarize
from helpers.input import function_from_list

programs = [("PubMed QA: Raw", pubmed_raw), ("PubMed QA: Summarized", pubmed_summarize)]

if __name__ == "__main__":
    function_from_list("What database would you like to run?", programs)
