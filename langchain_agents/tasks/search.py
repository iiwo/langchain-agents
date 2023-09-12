from langchain_agents.tasks.base_task import BaseTask
from langchain.tools import DuckDuckGoSearchResults
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_agents.tools.webpage_qa_tool import WebpageQATool

TASK = "What is the age of the captain of current, Group E, leading team in the 2024 UEFA euro qualifiers?"


class Search(BaseTask):
    NAME = 'search'

    def get_prompt(self):
        return TASK

    def get_tools(self, llm):
        return [
            DuckDuckGoSearchResults(),
            WebpageQATool(qa_chain=load_qa_with_sources_chain(llm=llm))
        ]
