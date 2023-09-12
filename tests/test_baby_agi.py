import logging
import langchain

from langchain_agents.tasks.search import Search
from langchain_agents.agents.baby_agi import BabyAGI

# log_level = logging.DEBUG
# logging.basicConfig(level=log_level)
# langchain.debug = True


def test_agent():
    agent = BabyAGI(task=Search())
    result = agent.run_task()
    print(result)
