from langchain_agents.agents.base_agent import BaseAgent
from langchain.chains.base import Chain
from langchain.agents import AgentType
from langchain.agents import initialize_agent


class ReAct(BaseAgent):
    NAME = 're_act'

    def build_chain(self) -> Chain:
        return initialize_agent(
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            llm=self.llm,
            tools=self.task.get_tools(llm=self.llm),
            max_iterations=15,
            verbose=True,
            handle_parsing_errors=True
        )
