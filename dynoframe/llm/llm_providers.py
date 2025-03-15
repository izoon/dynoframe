from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, 
                prompt: str,
                max_tokens: int = 1000,
                temperature: float = 0.7,
                top_p: float = 1.0,
                frequency_penalty: float = 0.0,
                presence_penalty: float = 0.0,
                stop_sequences: Optional[List[str]] = None,
                **kwargs) -> str:
        """
        Generate text based on the prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter (0-1)
            frequency_penalty: Penalty for token frequency (0-2)
            presence_penalty: Penalty for token presence (0-2)
            stop_sequences: List of sequences that stop generation
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available models for this provider.
        
        Returns:
            List of model identifiers
        """
        pass
    
    @abstractmethod
    def supports_streaming(self) -> bool:
        """
        Check if provider supports streaming responses.
        
        Returns:
            True if streaming is supported, False otherwise
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get the name of the provider.
        
        Returns:
            Provider name
        """
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        pass
    
    def validate_api_access(self) -> bool:
        """
        Validate API access by making a small test request.
        
        Returns:
            True if access is valid, False otherwise
        """
        try:
            _ = self.generate("Hello, world!", max_tokens=5, temperature=0.1)
            return True
        except Exception as e:
            logger.warning(f"API access validation failed: {str(e)}")
            return False


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation."""
    
    # Role-optimized model mapping
    ROLE_MODEL_MAPPING = {
        # Default mappings - GPT-3.5 for basic roles, GPT-4 for complex/specialized roles
        "default": "gpt-3.5-turbo",
        "researcher": "gpt-4-turbo",
        "expert": "gpt-4",
        "analyst": "gpt-4",
        "writer": "gpt-4",
        "assistant": "gpt-3.5-turbo",
        "translator": "gpt-3.5-turbo",
        "tutor": "gpt-4",
        "summarizer": "gpt-3.5-turbo",
        # Task-specific mappings
        "code_assistant": "gpt-4-turbo",
        "medical_expert": "gpt-4",
        "legal_expert": "gpt-4",
        "data_analyst": "gpt-4-turbo",
        "creative_writer": "gpt-4",
    }
    
    # Complexity-adjusted parameter presets
    COMPLEXITY_PRESETS = {
        "simple": {
            "temperature": 0.3,
            "max_tokens": 500,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        },
        "medium": {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.95,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        },
        "complex": {
            "temperature": 0.8,
            "max_tokens": 2000,
            "top_p": 1.0,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.2
        },
        "creative": {
            "temperature": 0.9,
            "max_tokens": 1500,
            "top_p": 1.0,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.1
        },
        "precise": {
            "temperature": 0.2,
            "max_tokens": 800,
            "top_p": 0.8,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    }
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 default_model: str = "gpt-3.5-turbo",
                 organization: Optional[str] = None):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            default_model: Default model to use
            organization: OpenAI organization ID
        """
        import openai
        self.default_model = default_model
        
        # Set API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OpenAI API key not provided. Please set OPENAI_API_KEY environment variable.")
        
        # Set client configuration
        openai.api_key = self.api_key
        if organization:
            openai.organization = organization
        
        self.client = openai
        
        # Import tiktoken for token estimation if available
        try:
            import tiktoken
            self.tokenizer = tiktoken.encoding_for_model(self.default_model)
        except (ImportError, KeyError):
            logger.warning("Tiktoken not available or model not supported. Token estimation will be approximate.")
            self.tokenizer = None
    
    def generate(self, 
                prompt: str,
                model: Optional[str] = None,
                max_tokens: int = 1000,
                temperature: float = 0.7,
                top_p: float = 1.0,
                frequency_penalty: float = 0.0,
                presence_penalty: float = 0.0,
                stop_sequences: Optional[List[str]] = None,
                **kwargs) -> str:
        """Generate text using OpenAI API."""
        try:
            model = model or self.default_model
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop_sequences,
                **kwargs
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def get_available_models(self) -> List[str]:
        """Get available OpenAI models."""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Error getting OpenAI models: {str(e)}")
            # Return common models as fallback
            return [
                "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-3.5-turbo-16k",
                "gpt-4-1106-preview", "gpt-4-vision-preview"
            ]
    
    def supports_streaming(self) -> bool:
        """Check if OpenAI supports streaming."""
        return True
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return "OpenAI"
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens for text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimation: 1 token ≈ 4 characters for English text
            return len(text) // 4
    
    def get_model_for_role(self, role: str) -> str:
        """
        Get the most appropriate model for a given role.
        
        Args:
            role: Agent role (e.g., "researcher", "assistant")
            
        Returns:
            Model identifier
        """
        role_lower = role.lower()
        
        # Check for exact matches
        if role_lower in self.ROLE_MODEL_MAPPING:
            return self.ROLE_MODEL_MAPPING[role_lower]
        
        # Check for partial matches
        for role_key in self.ROLE_MODEL_MAPPING:
            if role_key in role_lower:
                return self.ROLE_MODEL_MAPPING[role_key]
        
        # Default model
        return self.ROLE_MODEL_MAPPING["default"]
    
    def get_parameters_for_complexity(self, complexity: str) -> Dict[str, Any]:
        """
        Get generation parameters based on task complexity.
        
        Args:
            complexity: Task complexity level ("simple", "medium", "complex", "creative", "precise")
            
        Returns:
            Dictionary of parameters
        """
        if complexity.lower() in self.COMPLEXITY_PRESETS:
            return self.COMPLEXITY_PRESETS[complexity.lower()]
        else:
            # Default to medium complexity
            return self.COMPLEXITY_PRESETS["medium"]


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) LLM provider implementation."""
    
    # Role-optimized model mapping
    ROLE_MODEL_MAPPING = {
        "default": "claude-2",
        "researcher": "claude-2",
        "expert": "claude-2",
        "analyst": "claude-2",
        "writer": "claude-2",
        "assistant": "claude-instant-1",
        "translator": "claude-instant-1",
        "tutor": "claude-2",
        "summarizer": "claude-instant-1",
    }
    
    # Complexity-adjusted parameter presets
    COMPLEXITY_PRESETS = {
        "simple": {
            "temperature": 0.3,
            "max_tokens": 500,
            "top_p": 0.9,
        },
        "medium": {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.95,
        },
        "complex": {
            "temperature": 0.8,
            "max_tokens": 2000,
            "top_p": 1.0,
        },
        "creative": {
            "temperature": 0.9,
            "max_tokens": 1500,
            "top_p": 1.0,
        },
        "precise": {
            "temperature": 0.2,
            "max_tokens": 800,
            "top_p": 0.8,
        }
    }
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 default_model: str = "claude-2"):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            default_model: Default model to use
        """
        self.default_model = default_model
        
        # Set API key
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("Anthropic API key not provided. Please set ANTHROPIC_API_KEY environment variable.")
        
        # Import anthropic if available
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.anthropic = anthropic
            self._available = True
        except ImportError:
            logger.warning("Anthropic package not installed. Please install with: pip install anthropic")
            self._available = False
    
    def generate(self, 
                prompt: str,
                model: Optional[str] = None,
                max_tokens: int = 1000,
                temperature: float = 0.7,
                top_p: float = 1.0,
                frequency_penalty: float = 0.0,
                presence_penalty: float = 0.0,
                stop_sequences: Optional[List[str]] = None,
                **kwargs) -> str:
        """Generate text using Anthropic API."""
        if not self._available:
            return "Error: Anthropic package not installed"
        
        try:
            model = model or self.default_model
            
            # Anthropic expects a specific format
            formatted_prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
            
            response = self.client.completions.create(
                prompt=formatted_prompt,
                model=model,
                max_tokens_to_sample=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_sequences or ["\n\nHuman:"],
                **kwargs
            )
            
            return response.completion
        except Exception as e:
            logger.error(f"Error generating text with Anthropic: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def get_available_models(self) -> List[str]:
        """Get available Anthropic models."""
        if not self._available:
            return []
        
        # Anthropic doesn't have a models.list endpoint, so we return the known models
        return ["claude-2", "claude-instant-1"]
    
    def supports_streaming(self) -> bool:
        """Check if Anthropic supports streaming."""
        return True
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return "Anthropic"
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens for text."""
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    def get_model_for_role(self, role: str) -> str:
        """Get the most appropriate model for a given role."""
        role_lower = role.lower()
        
        # Check for exact matches
        if role_lower in self.ROLE_MODEL_MAPPING:
            return self.ROLE_MODEL_MAPPING[role_lower]
        
        # Check for partial matches
        for role_key in self.ROLE_MODEL_MAPPING:
            if role_key in role_lower:
                return self.ROLE_MODEL_MAPPING[role_key]
        
        # Default model
        return self.ROLE_MODEL_MAPPING["default"]
    
    def get_parameters_for_complexity(self, complexity: str) -> Dict[str, Any]:
        """Get generation parameters based on task complexity."""
        if complexity.lower() in self.COMPLEXITY_PRESETS:
            return self.COMPLEXITY_PRESETS[complexity.lower()]
        else:
            # Default to medium complexity
            return self.COMPLEXITY_PRESETS["medium"]


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    @staticmethod
    def create_provider(provider_name: str, **kwargs) -> LLMProvider:
        """
        Create an LLM provider instance.
        
        Args:
            provider_name: Name of the provider ("openai", "anthropic")
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLM provider instance
        
        Raises:
            ValueError: If provider is not supported
        """
        provider_name = provider_name.lower()
        
        if provider_name == "openai":
            return OpenAIProvider(**kwargs)
        elif provider_name == "anthropic":
            return AnthropicProvider(**kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")
    
    @staticmethod
    def get_default_provider() -> LLMProvider:
        """
        Get default provider based on available API keys.
        
        Returns:
            Default LLM provider
        """
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAIProvider()
        elif os.environ.get("ANTHROPIC_API_KEY"):
            return AnthropicProvider()
        else:
            logger.warning("No API keys found. Using OpenAI provider, but API calls will fail without a key.")
            return OpenAIProvider()


# Task complexity analyzer
class TaskComplexityAnalyzer:
    """Analyzer for determining task complexity."""
    
    # Keywords that indicate complexity
    COMPLEXITY_INDICATORS = {
        "simple": [
            "simple", "basic", "straightforward", "easy", "brief", 
            "short", "quick", "elementary", "fundamental"
        ],
        "medium": [
            "moderate", "standard", "normal", "regular", "ordinary",
            "intermediate", "average", "typical"
        ],
        "complex": [
            "complex", "complicated", "advanced", "sophisticated", "intricate",
            "detailed", "in-depth", "comprehensive", "extensive", "thorough"
        ],
        "creative": [
            "creative", "innovative", "original", "imaginative", "artistic",
            "novel", "unique", "inventive", "story", "narrative", "metaphor"
        ],
        "precise": [
            "precise", "exact", "accurate", "specific", "technical",
            "scientific", "mathematical", "factual", "logical", "code"
        ]
    }
    
    @staticmethod
    def analyze_complexity(query: str, role: str = None) -> str:
        """
        Analyze the complexity of a task based on the query and role.
        
        Args:
            query: User query
            role: Agent role (optional)
            
        Returns:
            Complexity level ("simple", "medium", "complex", "creative", "precise")
        """
        query_lower = query.lower()
        
        # Count complexity indicators for each level
        scores = {level: 0 for level in TaskComplexityAnalyzer.COMPLEXITY_INDICATORS}
        
        for level, indicators in TaskComplexityAnalyzer.COMPLEXITY_INDICATORS.items():
            for indicator in indicators:
                if indicator in query_lower:
                    scores[level] += 1
        
        # Apply role-based adjustments
        if role:
            role_lower = role.lower()
            if any(term in role_lower for term in ["researcher", "analyst", "expert"]):
                scores["complex"] += 1
            elif any(term in role_lower for term in ["creative", "writer", "artist"]):
                scores["creative"] += 1
            elif any(term in role_lower for term in ["technical", "engineer", "developer"]):
                scores["precise"] += 1
        
        # Analyze query length - longer queries often indicate complexity
        word_count = len(query_lower.split())
        if word_count > 50:
            scores["complex"] += 1
        elif word_count < 10:
            scores["simple"] += 1
        
        # Check for question complexity
        question_words = ["how", "why", "what", "when", "where", "who", "which"]
        advanced_indicators = ["explain", "analyze", "compare", "contrast", "evaluate", "synthesize"]
        
        if any(query_lower.startswith(word + " ") for word in question_words):
            # Simple questions
            if any(word + " is" in query_lower for word in question_words):
                scores["simple"] += 1
            # More complex questions
            if any(indicator in query_lower for indicator in advanced_indicators):
                scores["complex"] += 1
        
        # Determine the highest score
        max_level = max(scores, key=scores.get)
        
        # If scores are tied or all zero, default to medium
        if list(scores.values()).count(scores[max_level]) > 1 or scores[max_level] == 0:
            return "medium"
        
        return max_level 