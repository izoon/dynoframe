import os
import json
from typing import Dict, Optional, Any

from .llm_providers import LLMProvider
from .openrouter_provider import OpenRouterProvider


class LLMProviderFactory:
    """Factory class to create and manage LLM provider instances."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the LLM provider factory.
        
        Args:
            config_path: Path to the config file
        """
        self.config_path = config_path
        self._providers: Dict[str, LLMProvider] = {}
        self._default_provider: Optional[str] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from the config file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
                
            # Set default provider from config if available
            if "default_provider" in self.config:
                self._default_provider = self.config["default_provider"]
            else:
                # Default to OpenRouter if available, otherwise none
                if os.environ.get("OPENROUTER_API_KEY") or self.config.get("api_keys", {}).get("openrouter"):
                    self._default_provider = "openrouter"
        except (FileNotFoundError, json.JSONDecodeError):
            self.config = {}
    
    def get_provider(self, provider_name: Optional[str] = None) -> LLMProvider:
        """Get a provider instance by name or the default provider.
        
        Args:
            provider_name: Name of the provider to get, or None for default
            
        Returns:
            An instance of the requested LLM provider
            
        Raises:
            ValueError: If the requested provider is not supported or no default is available
        """
        # Use default provider if none specified
        if provider_name is None:
            if self._default_provider is None:
                raise ValueError("No default provider configured. Please specify a provider.")
            provider_name = self._default_provider
        
        # Convert to lowercase for case-insensitive comparison
        provider_name = provider_name.lower()
        
        # Return cached provider if available
        if provider_name in self._providers:
            return self._providers[provider_name]
        
        # Create a new provider instance
        if provider_name == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY") or self.config.get("api_keys", {}).get("openrouter")
            provider = OpenRouterProvider(api_key=api_key, config_path=self.config_path)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
        
        # Cache the provider
        self._providers[provider_name] = provider
        return provider
    
    def get_provider_for_role(self, role: str) -> LLMProvider:
        """Get the appropriate provider for a specific role.
        
        Args:
            role: The role (e.g., "assistant", "researcher", "coder")
            
        Returns:
            An appropriate LLM provider instance for the role
        """
        # Get default provider for now - in future, could map roles to specific providers
        provider = self.get_provider()
        return provider
    
    def get_generation_params(self, complexity: str = "medium", role: str = "assistant") -> Dict[str, Any]:
        """Get generation parameters optimized for a specific complexity and role.
        
        Args:
            complexity: Task complexity ("simple", "medium", "complex", "creative")
            role: The role for model selection
            
        Returns:
            Dictionary of generation parameters
        """
        provider = self.get_provider_for_role(role)
        params = provider.get_parameters_for_complexity(complexity)
        
        # Add the role-specific model
        if hasattr(provider, "get_model_for_role"):
            model = provider.get_model_for_role(role)
            if model:
                params["model"] = model
        
        return params


# Usage example
if __name__ == "__main__":
    try:
        factory = LLMProviderFactory()
        provider = factory.get_provider()
        
        print(f"Using provider: {provider.get_provider_name()}")
        
        # Test with different complexity and roles
        simple_params = factory.get_generation_params("simple", "assistant")
        complex_params = factory.get_generation_params("complex", "researcher")
        
        print(f"Simple task parameters: {simple_params}")
        print(f"Complex research parameters: {complex_params}")
        
        # Generate a response
        prompt = "What is the capital of France?"
        response = provider.generate(prompt, **simple_params)
        
        print("\nGenerated response:")
        print(response)
        
    except Exception as e:
        print(f"Error: {str(e)}") 