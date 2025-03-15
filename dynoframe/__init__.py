"""
DynoFrame - A framework for building dynamic agent-based systems

DynoFrame provides a flexible framework for creating and orchestrating intelligent agents
that can perform complex tasks using LLMs and other AI technologies.
"""

__version__ = "0.1.0"

# Import core components
from .core.dyno_agent import DynoAgent
from .core.team import Team
from .core.dyno_agent_with_tools import DynoAgentWithTools
from .core.task_complexity import TaskComplexityAnalyzer

# Import RAG components
from .rag.rag_agent import DynoRAGAgent

# Import LLM providers
from .llm.llm_providers import LLMProvider
from .llm.llm_provider_factory import LLMProviderFactory
from .llm.openrouter_provider import OpenRouterProvider
from .llm.openrouter_config import OpenRouterConfig

# Import PDF processing
from .pdf.pdf_processing_agent import PDFProcessingDecisionAgent

# Import utility modules
from . import llama_index_compat
from . import dyno_llamaindex

# Export all key classes and functions
__all__ = [
    'DynoAgent',
    'Team',
    'DynoAgentWithTools',
    'DynoRAGAgent',
    'LLMProvider',
    'LLMProviderFactory',
    'OpenRouterProvider',
    'OpenRouterConfig',
    'TaskComplexityAnalyzer',
    'PDFProcessingDecisionAgent',
    'llama_index_compat',
    'dyno_llamaindex',
] 