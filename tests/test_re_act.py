import logging
import langchain

from langchain_agents.tasks.search import Search
from langchain_agents.agents.re_act import ReAct

# log_level = logging.DEBUG
# logging.basicConfig(level=log_level)
# langchain.debug = True


def test_agent():
    agent = ReAct(task=Search())
    result = agent.run_task()
    print(result)
