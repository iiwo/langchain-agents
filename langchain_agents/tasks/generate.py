from langchain_agents.tasks.base_task import BaseTask
from langchain.tools import DuckDuckGoSearchResults
from langchain.tools.requests.tool import RequestsGetTool
from langchain.utilities.requests import TextRequestsWrapper

TASK = """
"""


class Generate(BaseTask):
    def get_prompt(self):
        return TASK

    def get_tools(self):
        return [
            DuckDuckGoSearchResults(),
            RequestsGetTool(requests_wrapper=TextRequestsWrapper())
        ]
