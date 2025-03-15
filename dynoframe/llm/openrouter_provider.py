import os
import json
import requests
from typing import Dict, List, Any, Optional, Union

from openrouter_config import OpenRouterConfig
from llm_providers import LLMProvider

class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider implementation for DynoFrame."""
    
    def __init__(self, api_key: Optional[str] = None, config_path: str = "config.json"):
        """Initialize the OpenRouter provider.
        
        Args:
            api_key: OpenRouter API key (optional, will use config or env var if not provided)
            config_path: Path to the config file
        """
        self.config = OpenRouterConfig(config_path)
        
        # Use the provided API key if given, otherwise use the one from config
        if api_key:
            self.config.api_key = api_key
        
        # Check for API key validity
        if not self.config.api_key:
            raise ValueError("OpenRouter API key is required. Set it in config.json or the OPENROUTER_API_KEY environment variable.")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the OpenRouter API.
        
        Args:
            prompt: The text prompt
            **kwargs: Additional parameters to override defaults
        
        Returns:
            Generated text response
        """
        # Get default parameters from config
        params = self.config.get_default_generation_params()
        
        # Override with provided parameters
        for key, value in kwargs.items():
            if key in params:
                params[key] = value
        
        # Prepare the API request
        api_params = self.config.get_api_params()
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": params.get("model", self.config.get_default_generation_params()["model"]),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 1000)
        }
        
        # Add any additional parameters if provided
        if "top_p" in kwargs:
            data["top_p"] = kwargs["top_p"]
        if "frequency_penalty" in kwargs:
            data["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            data["presence_penalty"] = kwargs["presence_penalty"]
        
        # Make the API request
        try:
            url = f"{api_params['base_url']}/chat/completions"
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=api_params.get("timeout", 60)
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract the generated text
            return result["choices"][0]["message"]["content"]
        
        except requests.RequestException as e:
            # Try a fallback model if available
            if "model" in kwargs and kwargs["model"] != params["model"]:
                fallback_models = self.config.config["openrouter_config"]["fallback_models"]
                if fallback_models:
                    print(f"Error with model {params['model']}, trying fallback model {fallback_models[0]}")
                    kwargs["model"] = fallback_models[0]
                    return self.generate(prompt, **kwargs)
            
            # If no fallback or already tried, raise the error
            raise RuntimeError(f"Error generating text with OpenRouter: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """Get a list of available models from OpenRouter."""
        try:
            api_params = self.config.get_api_params()
            headers = {
                "Authorization": f"Bearer {self.config.api_key}"
            }
            
            response = requests.get(
                f"{api_params['base_url']}/models",
                headers=headers,
                timeout=api_params.get("timeout", 60)
            )
            
            response.raise_for_status()
            models = response.json()
            
            # Extract model IDs
            return [model["id"] for model in models["data"]]
        except:
            # Return default models from config if API call fails
            default_model = self.config.config["openrouter_config"]["default_model"]
            fallback_models = self.config.config["openrouter_config"]["fallback_models"]
            return [default_model] + fallback_models
    
    def supports_streaming(self) -> bool:
        """Check if streaming is supported."""
        return True
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "OpenRouter"
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in the text."""
        # Simple estimation: ~4 characters per token on average
        return len(text) // 4
    
    def get_model_for_role(self, role: str) -> str:
        """Get the preferred model for a given role."""
        return self.config.get_model_for_role(role)
    
    def get_parameters_for_complexity(self, complexity: str) -> Dict[str, Any]:
        """Get parameters adjusted for task complexity."""
        # Default parameters
        params = {
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        # Adjust based on complexity
        if complexity == "simple":
            params["temperature"] = 0.3
            params["max_tokens"] = 500
        elif complexity == "complex":
            params["temperature"] = 0.6
            params["max_tokens"] = 2000
        elif complexity == "creative":
            params["temperature"] = 0.9
            params["max_tokens"] = 1500
        
        return params


# Usage example
if __name__ == "__main__":
    try:
        provider = OpenRouterProvider()
        models = provider.get_available_models()
        print(f"Available models: {', '.join(models[:5])}...")
        
        # Test generation
        prompt = "Explain the concept of Retrieval-Augmented Generation in one paragraph."
        response = provider.generate(prompt, max_tokens=150)
        
        print("\nGenerated response:")
        print(response)
        
        # Test role-based model selection
        researcher_model = provider.get_model_for_role("researcher")
        print(f"\nModel for researcher role: {researcher_model}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please ensure your OpenRouter API key is set correctly.") 