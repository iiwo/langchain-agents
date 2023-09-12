import logging
import langchain

from langchain_agents.agents.plan_and_execute import PlanAndExecute
from langchain_agents.tasks.search import Search

# log_level = logging.DEBUG
# logging.basicConfig(level=log_level)
# langchain.debug = True


def test_agent():
    agent = PlanAndExecute(task=Search())
    result = agent.run_task()
    print(result)
