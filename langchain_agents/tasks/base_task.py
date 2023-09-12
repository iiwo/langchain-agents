from abc import ABC


class BaseTask(ABC):
    NAME = 'task'

    def get_prompt(self):
        raise NotImplementedError()

    def get_tools(self, llm):
        raise NotImplementedError()
