import time

from langchain_agents.agents.base_agent import BaseAgent
from langchain_experimental.autonomous_agents import AutoGPT as LangchainAutoGPT
from langchain.callbacks import get_openai_callback
from langchain_agents.agents.agent_response import AgentResponse


class AutoGPT(BaseAgent):
    NAME = 'auto_gpt'

    def build_chain(self):
        return LangchainAutoGPT.from_llm_and_tools(
            ai_name='Tom',
            ai_role='Assistant',
            tools=self.task.get_tools(llm=self.llm),
            llm=self.llm,
            memory=self.vectorstore().as_retriever(),
        )

    def run_task(self) -> AgentResponse:
        with get_openai_callback() as callback:
            start_time = time.perf_counter()
            output = self.chain.run([self.task.get_prompt()])
            end_time = time.perf_counter()

            return AgentResponse.build(
                output=output,
                open_ai_callback=callback,
                llm=self.llm,
                tools=self.task.get_tools(llm=self.llm),
                total_time=end_time - start_time
            ).dict()


