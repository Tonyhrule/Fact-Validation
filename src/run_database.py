from database.pubmed.raw import pubmed_raw
from database.pubmed.summarizer import pubmed_summarize
from database.squad.raw import squad_raw
from database.squad.summarizer import squad_summarize
from helpers.input import function_from_list

programs = [
    ("PubMed QA: Raw", pubmed_raw),
    ("PubMed QA: Summarized", pubmed_summarize),
    ("Sqaud: Raw", squad_raw),
    ("Squad: Summarized", squad_summarize),
]

if __name__ == "__main__":
    function_from_list("What database would you like to run?", programs)
