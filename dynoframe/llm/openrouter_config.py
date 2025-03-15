import json
import os
from typing import Dict, List, Any, Optional

class OpenRouterConfig:
    """Utility class for managing OpenRouter API configuration."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Check for environment variable override
        env_api_key = os.environ.get("OPENROUTER_API_KEY")
        if env_api_key:
            self.config["api_keys"]["openrouter"] = env_api_key
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from the JSON file."""
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Configuration file {self.config_path} not found. Using default configuration.")
            return self._get_default_config()
        except json.JSONDecodeError:
            print(f"Warning: Configuration file {self.config_path} is not valid JSON. Using default configuration.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if the file cannot be loaded."""
        return {
            "api_keys": {
                "openrouter": ""
            },
            "openrouter_config": {
                "base_url": "https://openrouter.ai/api/v1",
                "default_model": "anthropic/claude-3-opus",
                "fallback_models": ["openai/gpt-4-turbo", "anthropic/claude-3-sonnet"],
                "default_temperature": 0.7,
                "default_max_tokens": 1000,
                "http_timeout": 60
            },
            "model_preferences": {
                "researcher": "anthropic/claude-3-opus",
                "assistant": "anthropic/claude-3-haiku",
                "coder": "openai/gpt-4-turbo",
                "analyzer": "anthropic/claude-3-sonnet"
            }
        }
    
    @property
    def api_key(self) -> str:
        """Get the OpenRouter API key."""
        return self.config["api_keys"]["openrouter"]
    
    @api_key.setter
    def api_key(self, value: str) -> None:
        """Set the OpenRouter API key."""
        self.config["api_keys"]["openrouter"] = value
        self._save_config()
    
    def get_model_for_role(self, role: str) -> str:
        """Get the preferred model for a given role.
        
        Args:
            role: The agent role (e.g., 'researcher', 'assistant')
            
        Returns:
            Model identifier string
        """
        role = role.lower()
        return self.config["model_preferences"].get(
            role, self.config["openrouter_config"]["default_model"]
        )
    
    def get_api_params(self) -> Dict[str, Any]:
        """Get the API parameters for OpenRouter requests."""
        return {
            "base_url": self.config["openrouter_config"]["base_url"],
            "timeout": self.config["openrouter_config"]["http_timeout"]
        }
    
    def get_default_generation_params(self) -> Dict[str, Any]:
        """Get default parameters for text generation."""
        config = self.config["openrouter_config"]
        return {
            "model": config["default_model"],
            "temperature": config["default_temperature"],
            "max_tokens": config["default_max_tokens"]
        }
    
    def _save_config(self) -> None:
        """Save the configuration back to the JSON file."""
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=4)


# Usage example
if __name__ == "__main__":
    config = OpenRouterConfig()
    
    # Check if API key is set
    if not config.api_key:
        print("OpenRouter API key not found. Please set it in config.json or use the OPENROUTER_API_KEY environment variable.")
        api_key = input("Enter your OpenRouter API key: ")
        config.api_key = api_key
    
    print(f"Using OpenRouter with base URL: {config.get_api_params()['base_url']}")
    print(f"Default model: {config.get_default_generation_params()['model']}")
    print(f"Model for researcher role: {config.get_model_for_role('researcher')}")
    print(f"Model for assistant role: {config.get_model_for_role('assistant')}") 