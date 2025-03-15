# DynoFrame

A framework for building dynamic agent-based systems with Large Language Models.

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/izoon/dynoframe.git
```

## Features

- **Agent Framework**: Build LLM-powered agents with defined roles, skills, and goals
- **Team Orchestration**: Create teams of agents with explicit dependency management
- **RAG Integration**: Built-in Retrieval Augmented Generation capabilities
- **OpenRouter Integration**: Support for multiple LLM providers through OpenRouter
- **PDF Processing**: Specialized agents for document extraction and analysis
- **Dynamic Role-Based Model Selection**: Automatically select the best LLM for each task

## Basic Usage

### Creating an Agent

```python
from dynoframe import DynoAgent

agent = DynoAgent(
    name="MyAgent",
    role="assistant",
    skills=["analysis", "planning"],
    goal="Help users accomplish their tasks"
)

response = agent.act("What can you help me with?")
print(response)
```

### Creating a Team

```python
from dynoframe import DynoAgent, Team

agent1 = DynoAgent(name="Planner", role="planner")
agent2 = DynoAgent(name="Executor", role="executor")

team = Team(
    name="TaskTeam",
    agents=[agent1, agent2],
    explicit_dependencies={"Executor": ["Planner"]}
)

results = team.execute({"input": "Build a simple website"})
print(results)
```

### Using RAG Agent

```python
from dynoframe import DynoRAGAgent

rag_agent = DynoRAGAgent(
    name="ResearchAgent",
    role="researcher",
    provider_name="openrouter",
    use_role_based_model=True
)

# Index documents
rag_agent.index_documents("path/to/documents")

# Query with RAG
answer = rag_agent.answer_question("What do the documents say about X?")
print(answer)
```

## Advanced Usage

For advanced usage examples, see the [full documentation](https://github.com/izoon/dynoframe).

## License

MIT 