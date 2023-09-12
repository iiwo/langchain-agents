import time
import os

from abc import ABC
from dotenv import load_dotenv
from loguru import logger

from langchain.callbacks import FileCallbackHandler
from langchain.chains.base import Chain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain_agents.agents.agent_response import AgentResponse
from langchain_agents.tasks.base_task import BaseTask
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_plantuml import diagram

load_dotenv()

OPENAI_API_KEY = os.getenv("OPEN_AI_API_KEY")
MODEL_GPT_3 = "gpt-3.5-turbo"
MODEL_GPT_4 = "gpt-4"
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"


class BaseAgent(ABC):
    def __init__(self, task: BaseTask, model_name: str = MODEL_GPT_3):
        self.diagram = diagram.activity_diagram_callback(note_max_length=2000)
        self.model_name = model_name
        self.task = task
        self.llm = self.get_llm()
        self.chain = self.build_chain()

    def run_task(self) -> AgentResponse:
        with get_openai_callback() as callback:
            start_time = time.perf_counter()
            try:
                output = self.chain.run(input=self.task.get_prompt())
            finally:
                self.finalize()
            end_time = time.perf_counter()

            response = AgentResponse.build(
                output=output,
                open_ai_callback=callback,
                llm=self.llm,
                tools=self.task.get_tools(llm=self.llm),
                total_time=end_time - start_time
            ).dict()
            return response

    def build_chain(self) -> Chain:
        raise NotImplementedError()

    def get_llm(self) -> BaseChatModel:
        return ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            model=self.model_name,
            verbose=True,
            max_retries=3,
            max_tokens=200
        )

    @classmethod
    def embedding_function(cls):
        return OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model=EMBEDDING_MODEL_NAME
        )

    @classmethod
    def vectorstore(cls):
        return Chroma(
            persist_directory="./tmp/chroma_db",
            embedding_function=cls.embedding_function()
        )

    def get_callbacks(self):
        return [self.agent_logger()]

    def finalize(self):
        self.diagram.save_uml_content(f'./tmp/{self.name()}.puml')

    def name(self):
        return f"{self.NAME}_{self.task.NAME}"

    def agent_logger(self):
        logfile = f'./tmp/{self.name()}.log'
        logger.add(logfile, colorize=True, enqueue=True)
        return FileCallbackHandler(logfile)
