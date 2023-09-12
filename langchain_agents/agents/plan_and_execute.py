from langchain_agents.agents.base_agent import BaseAgent
from langchain.chains.base import Chain
from langchain_experimental.plan_and_execute import(
    PlanAndExecute as LangchainPlanAndExecute,
    load_agent_executor,
    load_chat_planner
)


class PlanAndExecute(BaseAgent):
    NAME = 'plan_and_execute'

    def build_chain(self) -> Chain:
        planner = load_chat_planner(self.llm)
        executor = load_agent_executor(self.llm, self.task.get_tools(llm=self.llm), verbose=True)
        return LangchainPlanAndExecute(
            planner=planner,
            executor=executor,
            verbose=True,
            callbacks=self.get_callbacks(),
            handle_parsing_errors=True
        )
