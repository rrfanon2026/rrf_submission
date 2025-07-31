"""Utility functions for LLM operations."""


from .base import BaseLLM
from .openai._openai import OpenAILLM
from .gemini._gemini import GeminiLLM
from .anthropic._anthropic import AnthropicLLM

def get_llm_client(provider: str, model_name: str, temperature: float) -> BaseLLM:
    """
    Get the appropriate LLM client based on the provider.
    
    Args:
        provider: Name of the provider (e.g., 'openai', 'gemini', 'anthropic')
        model_name: Name of the model to use
        temperature: Temperature setting for the model
        
    Returns:
        LLM client instance
    
    Raises:
        ValueError: If the provider is not supported
    """
    providers = {
        'openai': OpenAILLM,
        'gemini': GeminiLLM,
        'anthropic': AnthropicLLM,
    }
    
    if provider.lower() not in providers:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers are: {list(providers.keys())}")
    
    return providers[provider.lower()](model_name=model_name, temperature=temperature)