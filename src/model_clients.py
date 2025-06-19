"""
AI model clients for querying different frontier AI models.
"""
import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

import openai
import anthropic
import google.generativeai as genai
try:
    from .types import MaxDiffItem, TrialSet, ModelResponse, EngineConfig
except ImportError:
    from types import MaxDiffItem, TrialSet, ModelResponse, EngineConfig


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
        """Create a prompt for the MaxDiff trial."""
        items_list = "\\n".join([f"{i+1}. {item.name}" for i, item in enumerate(trial.items)])
        
        prompt = f"""You are participating in a MaxDiff (Maximum Difference Scaling) survey. 

{config.instruction_text}

Here are the items to evaluate:
{items_list}

Please respond with a JSON object containing:
1. "best_item": The number (1-{len(trial.items)}) of the item you find {config.dimension_positive_label.lower()}
2. "worst_item": The number (1-{len(trial.items)}) of the item you find {config.dimension_negative_label.lower()}  
3. "reasoning": A brief explanation of your choices

Example response format:
{{
    "best_item": 2,
    "worst_item": 4,
    "reasoning": "I chose item 2 as best because... and item 4 as worst because..."
}}

Respond only with the JSON object, no additional text."""

        return prompt
    
    def _parse_response(self, response_text: str, trial: TrialSet) -> Dict[str, Any]:
        """Parse the model response to extract best and worst item choices."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\\{.*\\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                
                best_idx = int(parsed.get('best_item', 0)) - 1
                worst_idx = int(parsed.get('worst_item', 0)) - 1
                
                if 0 <= best_idx < len(trial.items) and 0 <= worst_idx < len(trial.items):
                    return {
                        'best_item_id': trial.items[best_idx].id,
                        'worst_item_id': trial.items[worst_idx].id,
                        'reasoning': parsed.get('reasoning', ''),
                        'success': True
                    }
            
            # Fallback: try to extract numbers from text
            numbers = re.findall(r'\\b([1-9]\\d*)\\b', response_text)
            if len(numbers) >= 2:
                best_idx = int(numbers[0]) - 1
                worst_idx = int(numbers[1]) - 1
                
                if 0 <= best_idx < len(trial.items) and 0 <= worst_idx < len(trial.items):
                    return {
                        'best_item_id': trial.items[best_idx].id,
                        'worst_item_id': trial.items[worst_idx].id,
                        'reasoning': 'Extracted from text response',
                        'success': True
                    }
            
            return {
                'best_item_id': trial.items[0].id,
                'worst_item_id': trial.items[-1].id,
                'reasoning': 'Fallback: could not parse response',
                'success': False,
                'error_message': 'Could not parse model response'
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
    
    async def evaluate_trial(self, trial: TrialSet, config: EngineConfig) -> ModelResponse:
        """Evaluate a trial using OpenAI's API."""
        try:
            prompt = self._create_prompt(trial, config)
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that responds with JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content
            parsed = self._parse_response(response_text, trial)
            
            return ModelResponse(
                model_name=f"openai-{self.model_name}",
                trial_number=trial.trial_number,
                best_item_id=parsed['best_item_id'],
                worst_item_id=parsed['worst_item_id'],
                reasoning=parsed.get('reasoning'),
                raw_response=response_text,
                success=parsed.get('success', True),
                error_message=parsed.get('error_message')
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
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    async def evaluate_trial(self, trial: TrialSet, config: EngineConfig) -> ModelResponse:
        """Evaluate a trial using Anthropic's API."""
        try:
            prompt = self._create_prompt(trial, config)
            
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text
            parsed = self._parse_response(response_text, trial)
            
            return ModelResponse(
                model_name=f"anthropic-{self.model_name}",
                trial_number=trial.trial_number,
                best_item_id=parsed['best_item_id'],
                worst_item_id=parsed['worst_item_id'],
                reasoning=parsed.get('reasoning'),
                raw_response=response_text,
                success=parsed.get('success', True),
                error_message=parsed.get('error_message')
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
    """Client for Google Gemini models."""
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    async def evaluate_trial(self, trial: TrialSet, config: EngineConfig) -> ModelResponse:
        """Evaluate a trial using Google's Gemini API."""
        try:
            prompt = self._create_prompt(trial, config)
            
            # Google's API is not async, so we run it in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=500,
                    )
                )
            )
            
            response_text = response.text
            parsed = self._parse_response(response_text, trial)
            
            return ModelResponse(
                model_name=f"google-{self.model_name}",
                trial_number=trial.trial_number,
                best_item_id=parsed['best_item_id'],
                worst_item_id=parsed['worst_item_id'],
                reasoning=parsed.get('reasoning'),
                raw_response=response_text,
                success=parsed.get('success', True),
                error_message=parsed.get('error_message')
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

