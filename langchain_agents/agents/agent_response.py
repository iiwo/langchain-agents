from pydantic import BaseModel


class Stats(BaseModel):
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    successful_requests: int
    total_cost: float
    total_time: float

    @staticmethod
    def from_open_ai_callback(open_ai_callback, total_time):
        return Stats(
            total_tokens=open_ai_callback.total_tokens,
            prompt_tokens=open_ai_callback.prompt_tokens,
            completion_tokens=open_ai_callback.completion_tokens,
            successful_requests=open_ai_callback.successful_requests,
            total_cost=open_ai_callback.total_cost,
            total_time=total_time
        )


class Details(BaseModel):
    model_name: str
    temperature: float

    @staticmethod
    def from_llm(llm):
        return Details(
            model_name=llm.model_name,
            temperature=llm.temperature
        )


class Meta(BaseModel):
    stats: Stats
    details: Details
    tools: list[str]

    @staticmethod
    def build(open_ai_callback, llm, tools, total_time):
        return Meta(
            stats=Stats.from_open_ai_callback(open_ai_callback, total_time),
            details=Details.from_llm(llm),
            tools=[tool.name for tool in tools]
        )


class AgentResponse(BaseModel):
    output: str
    meta: Meta

    @staticmethod
    def build(output, open_ai_callback, llm, tools, total_time):
        return AgentResponse(
            output=output,
            meta=Meta.build(open_ai_callback, llm, tools, total_time)
        )
