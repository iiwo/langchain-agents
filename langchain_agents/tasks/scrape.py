from langchain_agents.tasks.base_task import BaseTask
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_agents.tools.webpage_qa_tool import WebpageQATool

TASK = """Extract Renewable energy installations data from https://www.paih.gov.pl/sectors/renewable_energy and
format is as JSON in format: { "type": TYPE, "quantity": QUANTITY, "power: POWER } ans save to results.json file.
"""


class Scrape(BaseTask):
    def get_prompt(self):
        return TASK

    def get_tools(self, llm):
        return [
            WebpageQATool(qa_chain=load_qa_with_sources_chain(llm=llm))
        ]
