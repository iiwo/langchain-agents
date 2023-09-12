## Langchain agents test

Sample code for testing the [Langchain agents](https://python.langchain.com/docs/modules/agents/) in different scenarios.

### Example

```commandline
cp .env.example .env
```

```commandline
poetry install
```

```python
from langchain_agents.agents.plan_and_execute import PlanAndExecute
from langchain_agents.tasks.search import Search

agent = PlanAndExecute(task=Search())
result = agent.run_task()
print(result)
```

### Using pytest

```commandline
poetry run pytest tests/test_plan_and_execute.py -s
```