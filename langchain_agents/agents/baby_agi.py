import time
import faiss

from langchain_agents.agents.base_agent import BaseAgent
from langchain.agents import ZeroShotAgent, AgentExecutor, Tool
from langchain import LLMChain, PromptTemplate
from langchain_experimental.autonomous_agents import BabyAGI as LangchainBabyAGI
from langchain.callbacks import get_openai_callback
from langchain_agents.agents.agent_response import AgentResponse
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore


class BabyAGI(BaseAgent):
    NAME = 'baby_agi'

    def build_chain(self):
        prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into
        account these previously completed tasks: {context}."""
        suffix = """Question: {task}
        {agent_scratchpad}"""

        todo_prompt = PromptTemplate.from_template(
            "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a "
            "todo list for this objective: {objective}"
        )
        todo_chain = LLMChain(llm=self.llm, prompt=todo_prompt)

        tools = [
            Tool(
                name="TODO",
                func=todo_chain.run,
                description="useful for when you need to come up with todo lists. Input: an objective to create a "
                            "todo list for. Output: a todo list for that objective. Please be very clear what the "
                            "objective is!",
            ),
        ] + self.task.get_tools(llm=self.llm)
        tool_names = [tool.name for tool in tools]
        prompt = ZeroShotAgent.create_prompt(
            tools=tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=['objective', 'task', 'context', 'agent_scratchpad'],
        )
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True
        )
        return LangchainBabyAGI.from_llm(
            llm=self.llm,
            vectorstore=self.vectorstore(),
            task_execution_chain=agent_executor,
            verbose=True,
            max_iterations=15,
        )

    def run_task(self) -> AgentResponse:
        with get_openai_callback() as callback:
            start_time = time.perf_counter()
            output = self.chain({'objective': self.task.get_prompt()})
            end_time = time.perf_counter()

            return AgentResponse.build(
                output=output,
                open_ai_callback=callback,
                llm=self.llm,
                tools=self.task.get_tools(llm=self.llm),
                total_time=end_time - start_time
            ).dict()

    @classmethod
    def vectorstore(cls):
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        return FAISS(cls.embedding_function().embed_query, index, InMemoryDocstore({}), {})
