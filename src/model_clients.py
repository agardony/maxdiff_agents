"""
AI model clients for querying different frontier AI models.
"""
import asyncio
import json
import re
import time
import random
from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod

import openai
import anthropic
from google import genai
import instructor
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

try:
    from .types import MaxDiffItem, TrialSet, ModelResponse, EngineConfig, ModelConfig
except ImportError:
    from src.types import MaxDiffItem, TrialSet, ModelResponse, EngineConfig, ModelConfig


class MaxDiffResponse(BaseModel):
    """Pydantic model for structured MaxDiff responses from LLMs."""
    best_item: int = Field(
        ..., 
        ge=1, 
        description="The number (1-based index) of the item you find best/most preferred"
    )
    worst_item: int = Field(
        ..., 
        ge=1, 
        description="The number (1-based index) of the item you find worst/least preferred"
    )
    reasoning: str = Field(
        ..., 
        min_length=10,
        max_length=1000,
        description="A brief explanation of your choices"
    )
    
    @field_validator('best_item', 'worst_item')
    @classmethod
    def validate_item_numbers(cls, v):
        if v < 1:
            raise ValueError('Item numbers must be 1 or greater')
        return v
    
    @model_validator(mode='after')
    def validate_different_items(self):
        if self.best_item == self.worst_item:
            raise ValueError('Best and worst items must be different')
        return self


# Note: pydantic_to_gemini_schema converter removed - new Google GenAI SDK supports Pydantic models directly


def retry_with_backoff(config: ModelConfig = None):
    """Decorator that implements exponential backoff retry logic for async functions."""
    # Use environment variables as fallback if no config provided
    import os
    max_retries = config.max_retries if config else int(os.getenv('MAX_RETRIES', 3))
    base_delay = config.retry_base_delay if config else float(os.getenv('RETRY_BASE_DELAY', 1.0))
    max_delay = config.retry_max_delay if config else float(os.getenv('RETRY_MAX_DELAY', 60.0))
    timeout = config.request_timeout if config else int(os.getenv('REQUEST_TIMEOUT', 30))
    
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    return await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=float(timeout)
                    )
                except (asyncio.TimeoutError, openai.APITimeoutError, 
                       anthropic.APITimeoutError, Exception) as e:
                    last_exception = e
                    
                    # Check if this is a timeout-related error
                    is_timeout = (
                        isinstance(e, asyncio.TimeoutError) or
                        isinstance(e, openai.APITimeoutError) or
                        (hasattr(anthropic, 'APITimeoutError') and isinstance(e, anthropic.APITimeoutError)) or
                        'timeout' in str(e).lower() or
                        'timed out' in str(e).lower()
                    )
                    
                    # Only retry on timeout errors
                    if not is_timeout or attempt == max_retries:
                        break
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, 0.1 * delay)  # Add up to 10% jitter
                    total_delay = delay + jitter
                    
                    print(f"⏰ Timeout on attempt {attempt + 1}/{max_retries + 1} (Error: {type(e).__name__}), retrying in {total_delay:.2f}s...")
                    await asyncio.sleep(total_delay)
            
            # If we get here, all retries failed
            raise last_exception
        
        return wrapper
    return decorator


class ModelClient(ABC):
    """Abstract base class for AI model clients."""
    
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
    
    @abstractmethod
    async def evaluate_trial(self, trial: TrialSet, config: EngineConfig) -> ModelResponse:
        """Evaluate a MaxDiff trial and return the model's choice."""
        pass
    
    def _create_prompt(self, trial: TrialSet, config: EngineConfig) -> str:
        """Create a prompt for the MaxDiff trial with Pydantic schema."""
        items_list = "\n".join([f"{i+1}. {item.name}" for i, item in enumerate(trial.items)])
        
        # Get the Pydantic schema for structured responses
        schema = MaxDiffResponse.model_json_schema()
        
        prompt = f"""{config.persona}. You are participating in a MaxDiff (Maximum Difference Scaling) survey. 

{config.instruction_text}

Here are the items to evaluate:
{items_list}

IMPORTANT: You must respond with valid JSON that exactly matches this schema:
{json.dumps(schema, indent=2)}

Constraints:
- best_item: Must be a number from 1 to {len(trial.items)} representing the {config.dimension_positive_label.lower()} item
- worst_item: Must be a number from 1 to {len(trial.items)} representing the {config.dimension_negative_label.lower()} item
- best_item and worst_item must be different numbers
- reasoning: Must be 10-1000 characters explaining your choices

Example response format:
{{
    "best_item": 2,
    "worst_item": 4,
    "reasoning": "I chose item 2 as best because its flavor profile is most appealing to me, while item 4 is worst because I find it too intense for my preferences."
}}

Respond only with the JSON object, no additional text."""
        
        return prompt
    
    def _create_openai_prompt(self, trial: TrialSet, config: EngineConfig) -> str:
        """Create a simplified prompt for OpenAI that relies on response_format for structure."""
        items_list = "\n".join([f"{i+1}. {item.name}" for i, item in enumerate(trial.items)])
        
        prompt = f"""{config.persona}. You are participating in a MaxDiff (Maximum Difference Scaling) survey. 

{config.instruction_text}

Here are the items to evaluate:
{items_list}

Please provide:
- best_item: The number (1-{len(trial.items)}) of the item you find {config.dimension_positive_label.lower()}
- worst_item: The number (1-{len(trial.items)}) of the item you find {config.dimension_negative_label.lower()}
- reasoning: A brief explanation (10-1000 characters) of your choices

Make sure best_item and worst_item are different numbers."""
        
        return prompt
    
    def _parse_response(self, response_text: str, trial: TrialSet) -> Dict[str, Any]:
        """Parse the model response using Pydantic for structured validation."""
        try:
            # Strategy 1: Try to extract and parse JSON using Pydantic
            json_patterns = [
                r'\{[^{}]*"best_item"[^{}]*"worst_item"[^{}]*\}',  # Specific pattern
                r'\{.*?"best_item".*?"worst_item".*?\}',  # Flexible pattern
                r'\{.*\}',  # Broad pattern
            ]
            
            for pattern in json_patterns:
                json_matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
                for json_str in json_matches:
                    try:
                        # Clean up common JSON formatting issues
                        cleaned_json = json_str.strip()
                        cleaned_json = re.sub(r',\s*}', '}', cleaned_json)  # Remove trailing commas
                        cleaned_json = re.sub(r',\s*]', ']', cleaned_json)
                        
                        # Parse JSON and validate with Pydantic
                        raw_data = json.loads(cleaned_json)
                        validated_response = MaxDiffResponse(**raw_data)
                        
                        # Validate item indices are within range
                        if (1 <= validated_response.best_item <= len(trial.items) and 
                            1 <= validated_response.worst_item <= len(trial.items)):
                            
                            best_idx = validated_response.best_item - 1
                            worst_idx = validated_response.worst_item - 1
                            
                            return {
                                'best_item_id': trial.items[best_idx].id,
                                'worst_item_id': trial.items[worst_idx].id,
                                'reasoning': validated_response.reasoning,
                                'success': True
                            }
                        
                    except (json.JSONDecodeError, ValidationError, ValueError) as e:
                        continue  # Try next pattern
            
            # Strategy 2: Fallback to text extraction for backwards compatibility
            # Look for structured patterns in text
            best_patterns = [
                r'"?best_item"?\s*:?\s*([1-9]\d*)',
                r'best[_\s]*(?:item|choice)?[_\s]*:?[_\s]*([1-9]\d*)',
                r'item[_\s]*([1-9]\d*)[_\s]*(?:is|as)?[_\s]*(?:the[_\s]*)?best',
            ]
            
            worst_patterns = [
                r'"?worst_item"?\s*:?\s*([1-9]\d*)',
                r'worst[_\s]*(?:item|choice)?[_\s]*:?[_\s]*([1-9]\d*)',
                r'item[_\s]*([1-9]\d*)[_\s]*(?:is|as)?[_\s]*(?:the[_\s]*)?worst',
            ]
            
            best_item = None
            worst_item = None
            
            for pattern in best_patterns:
                matches = re.findall(pattern, response_text, re.IGNORECASE)
                if matches:
                    best_item = int(matches[0])
                    break
            
            for pattern in worst_patterns:
                matches = re.findall(pattern, response_text, re.IGNORECASE)
                if matches:
                    worst_item = int(matches[0])
                    break
            
            if (best_item and worst_item and 
                1 <= best_item <= len(trial.items) and 
                1 <= worst_item <= len(trial.items) and
                best_item != worst_item):
                
                return {
                    'best_item_id': trial.items[best_item - 1].id,
                    'worst_item_id': trial.items[worst_item - 1].id,
                    'reasoning': 'Extracted from text patterns',
                    'success': True
                }
            
            # Strategy 3: Final fallback - use first and last items
            return {
                'best_item_id': trial.items[0].id,
                'worst_item_id': trial.items[-1].id,
                'reasoning': f'Parse failed: Could not extract valid response from "{response_text[:100]}..."',
                'success': False,
                'error_message': 'Could not parse model response with any strategy'
            }
            
        except Exception as e:
            return {
                'best_item_id': trial.items[0].id,
                'worst_item_id': trial.items[-1].id,
                'reasoning': f'Error parsing response: {str(e)}',
                'success': False,
                'error_message': str(e)
            }


class OpenAIClient(ModelClient):
    """Client for OpenAI models."""
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    @retry_with_backoff()
    async def _make_api_call(self, prompt: str, trial: TrialSet) -> MaxDiffResponse:
        """Make the actual API call with retry logic using Pydantic response format."""
        import os
        response = await self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format=MaxDiffResponse,
            temperature=float(os.getenv('LLM_TEMPERATURE', 0.8)),
            max_tokens=int(os.getenv('LLM_MAX_TOKENS', 500)),
            top_p=float(os.getenv('LLM_TOP_P', 0.9))
        )
        
        parsed_response = response.choices[0].message.parsed
        
        # Validate item indices are within range
        if (1 <= parsed_response.best_item <= len(trial.items) and 
            1 <= parsed_response.worst_item <= len(trial.items)):
            return parsed_response
        else:
            # If indices are out of range, raise an error to trigger fallback
            raise ValueError(f"Item indices out of range: best={parsed_response.best_item}, worst={parsed_response.worst_item}, max={len(trial.items)}")
    
    async def evaluate_trial(self, trial: TrialSet, config: EngineConfig) -> ModelResponse:
        """Evaluate a trial using OpenAI's API with native Pydantic support."""
        try:
            prompt = self._create_openai_prompt(trial, config)
            pydantic_response = await self._make_api_call(prompt, trial)
            
            best_idx = pydantic_response.best_item - 1
            worst_idx = pydantic_response.worst_item - 1
            
            return ModelResponse(
                model_name=f"openai-{self.model_name}",
                trial_number=trial.trial_number,
                best_item_id=trial.items[best_idx].id,
                worst_item_id=trial.items[worst_idx].id,
                reasoning=pydantic_response.reasoning,
                raw_response=f"Parsed Pydantic response: {pydantic_response.model_dump()}",
                success=True,
                error_message=None
            )
            
        except Exception as e:
            return ModelResponse(
                model_name=f"openai-{self.model_name}",
                trial_number=trial.trial_number,
                best_item_id=trial.items[0].id,
                worst_item_id=trial.items[-1].id,
                reasoning=f"Error: {str(e)}",
                success=False,
                error_message=str(e)
            )


class AnthropicClient(ModelClient):
    """Client for Anthropic Claude models."""
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        # Create the base Anthropic client
        base_client = anthropic.AsyncAnthropic(api_key=api_key)
        # Patch it with Instructor for structured outputs
        self.client = instructor.from_anthropic(base_client)
        # Keep a reference to the base client for fallback
        self.base_client = base_client
    
    def _create_anthropic_prompt(self, trial: TrialSet, config: EngineConfig) -> str:
        """Create a simplified prompt for Anthropic that relies on Instructor for structure."""
        items_list = "\n".join([f"{i+1}. {item.name}" for i, item in enumerate(trial.items)])
        
        prompt = f"""{config.persona}. You are participating in a MaxDiff (Maximum Difference Scaling) survey. 

{config.instruction_text}

Here are the items to evaluate:
{items_list}

Please provide:
- best_item: The number (1-{len(trial.items)}) of the item you find {config.dimension_positive_label.lower()}
- worst_item: The number (1-{len(trial.items)}) of the item you find {config.dimension_negative_label.lower()}
- reasoning: A brief explanation (10-1000 characters) of your choices

Make sure best_item and worst_item are different numbers."""
        
        return prompt
    
    @retry_with_backoff()
    async def _make_api_call_structured(self, prompt: str, trial: TrialSet) -> MaxDiffResponse:
        """Make the actual API call with retry logic using Instructor structured output."""
        import os
        
        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=int(os.getenv('LLM_MAX_TOKENS', 500)),
            temperature=float(os.getenv('LLM_TEMPERATURE', 0.8)),
            top_p=float(os.getenv('LLM_TOP_P', 0.9)),
            messages=[{"role": "user", "content": prompt}],
            response_model=MaxDiffResponse
        )
        
        # Validate item indices are within range
        if (1 <= response.best_item <= len(trial.items) and 
            1 <= response.worst_item <= len(trial.items)):
            return response
        else:
            # If indices are out of range, raise an error to trigger fallback
            raise ValueError(f"Item indices out of range: best={response.best_item}, worst={response.worst_item}, max={len(trial.items)}")
    
    @retry_with_backoff()
    async def _make_api_call(self, prompt: str) -> str:
        """Make the actual API call with retry logic (fallback method)."""
        import os
        response = await self.base_client.messages.create(
            model=self.model_name,
            max_tokens=int(os.getenv('LLM_MAX_TOKENS', 500)),
            temperature=float(os.getenv('LLM_TEMPERATURE', 0.8)),
            top_p=float(os.getenv('LLM_TOP_P', 0.9)),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    async def evaluate_trial(self, trial: TrialSet, config: EngineConfig) -> ModelResponse:
        """Evaluate a trial using Anthropic's API with Instructor structured output."""
        try:
            prompt = self._create_anthropic_prompt(trial, config)
            pydantic_response = await self._make_api_call_structured(prompt, trial)
            
            best_idx = pydantic_response.best_item - 1
            worst_idx = pydantic_response.worst_item - 1
            
            return ModelResponse(
                model_name=f"anthropic-{self.model_name}",
                trial_number=trial.trial_number,
                best_item_id=trial.items[best_idx].id,
                worst_item_id=trial.items[worst_idx].id,
                reasoning=pydantic_response.reasoning,
                raw_response=f"Parsed Instructor response: {pydantic_response.model_dump()}",
                success=True,
                error_message=None
            )
            
        except Exception as e:
            return ModelResponse(
                model_name=f"anthropic-{self.model_name}",
                trial_number=trial.trial_number,
                best_item_id=trial.items[0].id,
                worst_item_id=trial.items[-1].id,
                reasoning=f"Error: {str(e)}",
                success=False,
                error_message=str(e)
            )


class GoogleClient(ModelClient):
    """Client for Google Gemini models using the new google-genai SDK."""
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        self.client = genai.Client(api_key=api_key)
    
    def _create_gemini_prompt(self, trial: TrialSet, config: EngineConfig) -> str:
        """Create a simplified prompt for Gemini that relies on response_schema for structure."""
        items_list = "\n".join([f"{i+1}. {item.name}" for i, item in enumerate(trial.items)])
        
        prompt = f"""{config.persona}. You are participating in a MaxDiff (Maximum Difference Scaling) survey. 

{config.instruction_text}

Here are the items to evaluate:
{items_list}

Please provide:
- best_item: The number (1-{len(trial.items)}) of the item you find {config.dimension_positive_label.lower()}
- worst_item: The number (1-{len(trial.items)}) of the item you find {config.dimension_negative_label.lower()}
- reasoning: A brief explanation (10-1000 characters) of your choices

Make sure best_item and worst_item are different numbers."""
        
        return prompt
    
    @retry_with_backoff()
    async def _make_api_call_structured(self, prompt: str, trial: TrialSet) -> MaxDiffResponse:
        """Make the actual API call with retry logic using the new Gemini SDK with direct Pydantic support."""
        import os
        
        # Use the new Google GenAI SDK with direct Pydantic support
        # Google's API is not async, so we run it in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": MaxDiffResponse,
                    "temperature": float(os.getenv('LLM_TEMPERATURE', 0.8)),
                    "max_output_tokens": int(os.getenv('LLM_MAX_TOKENS', 500)),
                    "top_p": float(os.getenv('LLM_TOP_P', 0.9)),
                }
            )
        )
        
        # The new SDK should return parsed Pydantic objects directly
        if hasattr(response, 'parsed') and response.parsed:
            pydantic_response = response.parsed
        else:
            # Fallback to JSON parsing if parsed attribute is not available
            try:
                raw_data = json.loads(response.text)
                pydantic_response = MaxDiffResponse(**raw_data)
            except (json.JSONDecodeError, ValidationError) as e:
                raise ValueError(f"Failed to parse Gemini structured response: {e}. Raw response: {response.text}")
        
        # Validate item indices are within range
        if (1 <= pydantic_response.best_item <= len(trial.items) and 
            1 <= pydantic_response.worst_item <= len(trial.items)):
            return pydantic_response
        else:
            # If indices are out of range, raise an error to trigger fallback
            raise ValueError(f"Item indices out of range: best={pydantic_response.best_item}, worst={pydantic_response.worst_item}, max={len(trial.items)}")
    
    async def evaluate_trial(self, trial: TrialSet, config: EngineConfig) -> ModelResponse:
        """Evaluate a trial using Google's Gemini API with the new SDK and direct Pydantic support."""
        try:
            prompt = self._create_gemini_prompt(trial, config)
            pydantic_response = await self._make_api_call_structured(prompt, trial)
            
            best_idx = pydantic_response.best_item - 1
            worst_idx = pydantic_response.worst_item - 1
            
            return ModelResponse(
                model_name=f"google-{self.model_name}",
                trial_number=trial.trial_number,
                best_item_id=trial.items[best_idx].id,
                worst_item_id=trial.items[worst_idx].id,
                reasoning=pydantic_response.reasoning,
                raw_response=f"Parsed Gemini structured response: {pydantic_response.model_dump()}",
                success=True,
                error_message=None
            )
            
        except Exception as e:
            return ModelResponse(
                model_name=f"google-{self.model_name}",
                trial_number=trial.trial_number,
                best_item_id=trial.items[0].id,
                worst_item_id=trial.items[-1].id,
                reasoning=f"Error: {str(e)}",
                success=False,
                error_message=str(e)
            )

