"""
Configuration utilities for AI model parameters with validation and model-specific constraints.
"""
import os
import warnings
from typing import Dict, Tuple, Union, Set


class TemperatureValidator:
    """Validates and clamps temperature values for different AI model providers."""
    
    # Model provider temperature ranges
    TEMPERATURE_RANGES = {
        'openai': (0.0, 2.0),
        'google': (0.0, 2.0), 
        'anthropic': (0.0, 1.0)
    }
    
    # Track what we've already informed about to avoid duplicates
    _warned_clamps: Set[Tuple[float, str]] = set()
    
    @classmethod
    def validate_and_clamp_temperature(cls, raw_temperature: Union[str, float], provider: str) -> float:
        """
        Validates and clamps temperature value for a specific model provider.
        
        Args:
            raw_temperature: Temperature value from environment or config
            provider: Model provider ('openai', 'google', 'anthropic')
            
        Returns:
            float: Validated and clamped temperature value
            
        Raises:
            ValueError: If temperature is outside valid range (0.0-2.0)
        """
        # Convert to float
        try:
            temperature = float(raw_temperature)
        except (ValueError, TypeError):
            raise ValueError(f"Temperature must be a valid number, got: {raw_temperature}")
        
        # Validate global range (0.0-2.0)
        if not (0.0 <= temperature <= 2.0):
            raise ValueError(
                f"Temperature must be between 0.0 and 2.0, got: {temperature}. "
                "Please update your .env file with a valid LLM_TEMPERATURE value."
            )
        
        # Get provider-specific range
        if provider not in cls.TEMPERATURE_RANGES:
            raise ValueError(f"Unknown provider: {provider}. Supported providers: {list(cls.TEMPERATURE_RANGES.keys())}")
        
        min_temp, max_temp = cls.TEMPERATURE_RANGES[provider]
        
        # Clamp to provider-specific range
        original_temperature = temperature
        temperature = max(min_temp, min(temperature, max_temp))
        
        # Inform about clamping if it occurred and hasn't been informed already
        if temperature != original_temperature:
            key = (original_temperature, provider)
            if key not in cls._warned_clamps:
                print(
                    f"Temperature {original_temperature} adjusted to {temperature} for {provider} "
                    f"(valid range: {min_temp}-{max_temp})."
                )
                cls._warned_clamps.add(key)
        
        return temperature
    
    @classmethod
    def get_temperature_for_provider(cls, provider: str) -> float:
        """
        Gets validated temperature for a specific provider from environment variables.
        
        Args:
            provider: Model provider ('openai', 'google', 'anthropic')
            
        Returns:
            float: Validated and clamped temperature value
        """
        raw_temperature = os.getenv('LLM_TEMPERATURE', '0.8')
        return cls.validate_and_clamp_temperature(raw_temperature, provider)


class ModelConfigValidator:
    """Validates other model configuration parameters."""
    
    @staticmethod
    def validate_max_tokens(raw_max_tokens: Union[str, int]) -> int:
        """Validate max_tokens parameter."""
        try:
            max_tokens = int(raw_max_tokens)
        except (ValueError, TypeError):
            raise ValueError(f"max_tokens must be a valid integer, got: {raw_max_tokens}")
        
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got: {max_tokens}")
        
        if max_tokens > 8192:  # Reasonable upper bound
            warnings.warn(
                f"max_tokens {max_tokens} is very high and may cause issues with some models",
                UserWarning
            )
        
        return max_tokens
    
    @staticmethod
    def validate_top_p(raw_top_p: Union[str, float]) -> float:
        """Validate top_p parameter."""
        try:
            top_p = float(raw_top_p)
        except (ValueError, TypeError):
            raise ValueError(f"top_p must be a valid number, got: {raw_top_p}")
        
        if not (0.0 <= top_p <= 1.0):
            raise ValueError(f"top_p must be between 0.0 and 1.0, got: {top_p}")
        
        return top_p


def get_validated_model_params(provider: str) -> Dict[str, Union[float, int]]:
    """
    Get all validated model parameters for a specific provider.
    
    Args:
        provider: Model provider ('openai', 'google', 'anthropic')
        
    Returns:
        Dict containing validated parameters: temperature, max_tokens, top_p
    """
    return {
        'temperature': TemperatureValidator.get_temperature_for_provider(provider),
        'max_tokens': ModelConfigValidator.validate_max_tokens(os.getenv('LLM_MAX_TOKENS', '500')),
        'top_p': ModelConfigValidator.validate_top_p(os.getenv('LLM_TOP_P', '0.9'))
    }


# Convenience functions for backward compatibility
def get_openai_temperature() -> float:
    """Get validated temperature for OpenAI models."""
    return TemperatureValidator.get_temperature_for_provider('openai')


def get_anthropic_temperature() -> float:
    """Get validated temperature for Anthropic models."""
    return TemperatureValidator.get_temperature_for_provider('anthropic')


def get_google_temperature() -> float:
    """Get validated temperature for Google models."""
    return TemperatureValidator.get_temperature_for_provider('google')
