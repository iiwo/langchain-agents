from langchain_agents.agents.base_agent import BaseAgent
from langchain.chains.base import Chain
from langchain.agents import AgentType
from langchain.agents import initialize_agent


class OpenAIFunctions(BaseAgent):
    NAME = 're_act'

    def build_chain(self) -> Chain:
        return initialize_agent(
            agent=AgentType.OPENAI_FUNCTIONS,
            llm=self.llm,
            tools=self.task.get_tools(llm=self.llm),
            max_iterations=15,
            verbose=True,
            handle_parsing_errors=True
        )
