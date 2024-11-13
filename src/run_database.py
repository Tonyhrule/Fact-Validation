from database.pubmed.raw import pubmed_raw
from helpers.input import function_from_list
import asyncio

programs = [("PubMed QA: Raw", pubmed_raw)]

if __name__ == "__main__":
    function_from_list("What database would you like to run?", programs)
