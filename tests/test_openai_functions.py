import logging
import langchain

from langchain_agents.tasks.search import Search
from langchain_agents.agents.openai_functions import OpenAIFunctions

# log_level = logging.DEBUG
# logging.basicConfig(level=log_level)
# langchain.debug = True


def test_agent():
    agent = OpenAIFunctions(task=Search())
    result = agent.run_task()
    print(result)
