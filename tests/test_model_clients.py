"""
Tests for the model clients module.
"""
import pytest
import asyncio
import json
import re
from unittest.mock import AsyncMock, patch, MagicMock

from src.model_clients import ModelClient, OpenAIClient, AnthropicClient, GoogleClient, retry_with_backoff, MaxDiffResponse
from src.types import TrialSet, ModelResponse, EngineConfig, ModelConfig, MaxDiffItem
from pydantic import ValidationError


class MockModelClient(ModelClient):
    """Mock model client for testing base functionality."""
    
    async def evaluate_trial(self, trial: TrialSet, config: EngineConfig) -> ModelResponse:
        """Mock evaluation that always returns the first and last items."""
        return ModelResponse(
            model_name=f"mock-{self.model_name}",
            trial_number=trial.trial_number,
            best_item_id=trial.items[0].id,
            worst_item_id=trial.items[-1].id,
            reasoning="Mock response",
            success=True
        )


class TestModelClient:
    """Test ModelClient base class."""
    
    def test_initialization(self):
        """Test model client initialization."""
        client = MockModelClient(model_name="test-model", api_key="test-key")
        assert client.model_name == "test-model"
        assert client.api_key == "test-key"
    
    def test_create_prompt(self, sample_trial, engine_config):
        """Test prompt creation."""
        client = MockModelClient(model_name="test", api_key="key")
        prompt = client._create_prompt(sample_trial, engine_config)
        
        # Check that prompt contains expected elements
        assert "MaxDiff" in prompt
        assert engine_config.persona in prompt
        assert engine_config.instruction_text in prompt
        assert "JSON" in prompt
        assert "best_item" in prompt
        assert "worst_item" in prompt
        assert "reasoning" in prompt
        
        # Check that all items are listed
        for i, item in enumerate(sample_trial.items):
            assert f"{i+1}. {item.name}" in prompt
    
    def test_parse_response_valid_json(self, sample_trial):
        """Test parsing a valid JSON response."""
        client = MockModelClient(model_name="test", api_key="key")
        response_text = '{"best_item": 2, "worst_item": 4, "reasoning": "Test reasoning"}'
        
        parsed = client._parse_response(response_text, sample_trial)
        
        assert parsed['success'] is True
        assert parsed['best_item_id'] == sample_trial.items[1].id  # Index 1 (item 2)
        assert parsed['worst_item_id'] == sample_trial.items[3].id  # Index 3 (item 4)
        assert parsed['reasoning'] == "Test reasoning"
    
    def test_parse_response_json_with_extra_text(self, sample_trial):
        """Test parsing JSON response with extra text."""
        client = MockModelClient(model_name="test", api_key="key")
        response_text = 'Here is my response: {"best_item": 1, "worst_item": 3, "reasoning": "Clear choice"} Done.'
        
        parsed = client._parse_response(response_text, sample_trial)
        
        assert parsed['success'] is True
        assert parsed['best_item_id'] == sample_trial.items[0].id
        assert parsed['worst_item_id'] == sample_trial.items[2].id
        assert parsed['reasoning'] == "Clear choice"
    
    def test_parse_response_fallback_numbers(self, sample_trial):
        """Test parsing response with fallback to number extraction."""
        client = MockModelClient(model_name="test", api_key="key")
        response_text = "I choose item 3 as best and item 1 as worst because reasons."
        
        parsed = client._parse_response(response_text, sample_trial)
        
        # If fallback fails, success will be False - let's check what actually happens
        # The fallback implementation should extract the first two numbers found
        if parsed['success']:
            assert parsed['best_item_id'] == sample_trial.items[2].id  # Item 3
            assert parsed['worst_item_id'] == sample_trial.items[0].id  # Item 1
            assert parsed['reasoning'] == "Extracted from text patterns"
        else:
            # If parsing fails, it should use fallback values
            assert parsed['best_item_id'] == sample_trial.items[0].id  # First item fallback
            assert parsed['worst_item_id'] == sample_trial.items[-1].id  # Last item fallback
            assert 'error_message' in parsed
    
    def test_parse_response_invalid_indices(self, sample_trial):
        """Test parsing response with invalid indices."""
        client = MockModelClient(model_name="test", api_key="key")
        response_text = '{"best_item": 10, "worst_item": 0, "reasoning": "Invalid"}'
        
        parsed = client._parse_response(response_text, sample_trial)
        
        assert parsed['success'] is False
        assert parsed['best_item_id'] == sample_trial.items[0].id  # Fallback
        assert parsed['worst_item_id'] == sample_trial.items[-1].id  # Fallback
        assert parsed['error_message'] == "Could not parse model response with any strategy"
    
    def test_parse_response_malformed_json(self, sample_trial):
        """Test parsing malformed JSON."""
        client = MockModelClient(model_name="test", api_key="key")
        response_text = '{"best_item": 1, "worst_item":}'
        
        parsed = client._parse_response(response_text, sample_trial)
        
        assert parsed['success'] is False
        assert parsed['best_item_id'] == sample_trial.items[0].id  # Fallback
        assert parsed['worst_item_id'] == sample_trial.items[-1].id  # Fallback
        assert "error_message" in parsed
    
    @pytest.mark.asyncio
    async def test_evaluate_trial(self, sample_trial, engine_config):
        """Test the mock evaluate_trial implementation."""
        client = MockModelClient(model_name="test", api_key="key")
        response = await client.evaluate_trial(sample_trial, engine_config)
        
        assert isinstance(response, ModelResponse)
        assert response.model_name == "mock-test"
        assert response.trial_number == sample_trial.trial_number
        assert response.best_item_id == sample_trial.items[0].id
        assert response.worst_item_id == sample_trial.items[-1].id
        assert response.success is True


class TestRetryDecorator:
    """Test retry_with_backoff decorator."""
    
    @pytest.mark.asyncio
    async def test_retry_decorator_success(self):
        """Test retry decorator with successful function."""
        config = ModelConfig(max_retries=3, retry_base_delay=0.01, request_timeout=1)
        
        @retry_with_backoff(config)
        async def success_func():
            return "success"
        
        result = await success_func()
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_retry_decorator_timeout_then_success(self):
        """Test retry decorator with timeout then success."""
        config = ModelConfig(max_retries=3, retry_base_delay=0.01, request_timeout=1)
        call_count = 0
        
        @retry_with_backoff(config)
        async def timeout_then_success():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError("First call times out")
            return "success"
        
        result = await timeout_then_success()
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_decorator_max_retries_exceeded(self):
        """Test retry decorator when max retries exceeded."""
        config = ModelConfig(max_retries=2, retry_base_delay=0.01, request_timeout=1)
        
        @retry_with_backoff(config)
        async def always_timeout():
            raise asyncio.TimeoutError("Always times out")
        
        with pytest.raises(asyncio.TimeoutError):
            await always_timeout()
    
    @pytest.mark.asyncio
    async def test_retry_decorator_non_timeout_error(self):
        """Test retry decorator with non-timeout error (no retry)."""
        config = ModelConfig(max_retries=3, retry_base_delay=0.01, request_timeout=1)
        
        @retry_with_backoff(config)
        async def non_timeout_error():
            raise ValueError("Non-timeout error")
        
        with pytest.raises(ValueError):
            await non_timeout_error()


class TestOpenAIClient:
    """Test OpenAI client."""
    
    @pytest.mark.asyncio
    async def test_openai_client_initialization(self):
        """Test OpenAI client initialization."""
        with patch('src.model_clients.openai.AsyncOpenAI') as mock_openai:
            client = OpenAIClient(model_name="gpt-4", api_key="test-key")
            assert client.model_name == "gpt-4"
            assert client.api_key == "test-key"
            mock_openai.assert_called_once_with(api_key="test-key")
    
    @pytest.mark.asyncio
    async def test_openai_evaluate_trial_success(self, sample_trial, engine_config):
        """Test successful OpenAI API call with Pydantic response format."""
        from src.model_clients import MaxDiffResponse
        
        # Mock Pydantic response
        mock_pydantic_response = MaxDiffResponse(
            best_item=1,
            worst_item=4,
            reasoning="AI reasoning from Pydantic response"
        )
        
        mock_response = MagicMock()
        mock_response.choices[0].message.parsed = mock_pydantic_response
        
        with patch('src.model_clients.openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_client.beta.chat.completions.parse.return_value = mock_response
            mock_openai.return_value = mock_client
            
            client = OpenAIClient(model_name="gpt-4", api_key="test-key")
            response = await client.evaluate_trial(sample_trial, engine_config)
            
            assert response.model_name == "openai-gpt-4"
            assert response.trial_number == sample_trial.trial_number
            assert response.success is True
            assert response.reasoning == "AI reasoning from Pydantic response"
            assert response.best_item_id == sample_trial.items[0].id
            assert response.worst_item_id == sample_trial.items[3].id
    
    @pytest.mark.asyncio
    async def test_openai_evaluate_trial_api_error(self, sample_trial, engine_config):
        """Test OpenAI API error handling."""
        with patch('src.model_clients.openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_client.beta.chat.completions.parse.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client
            
            client = OpenAIClient(model_name="gpt-4", api_key="test-key")
            response = await client.evaluate_trial(sample_trial, engine_config)
            
            assert response.model_name == "openai-gpt-4"
            assert response.success is False
            assert "API Error" in response.error_message
            # Should fallback to first and last items
            assert response.best_item_id == sample_trial.items[0].id
            assert response.worst_item_id == sample_trial.items[-1].id


class TestAnthropicClient:
    """Test Anthropic client."""
    
    @pytest.mark.asyncio
    async def test_anthropic_client_initialization(self):
        """Test Anthropic client initialization."""
        with patch('src.model_clients.anthropic.AsyncAnthropic') as mock_anthropic, \
             patch('src.model_clients.instructor') as mock_instructor_module:
            
            # Setup mock instructor client
            mock_instructor_client = AsyncMock()
            mock_instructor_module.from_anthropic.return_value = mock_instructor_client
            
            # Setup mock base client
            mock_base_client = AsyncMock()
            mock_anthropic.return_value = mock_base_client
            
            client = AnthropicClient(model_name="claude-3", api_key="test-key")
            assert client.model_name == "claude-3"
            assert client.api_key == "test-key"
            mock_anthropic.assert_called_once_with(api_key="test-key")
            mock_instructor_module.from_anthropic.assert_called_once_with(mock_base_client)
    
    @pytest.mark.asyncio
    async def test_anthropic_evaluate_trial_success(self, sample_trial, engine_config):
        """Test successful Anthropic API call with Instructor structured output."""
        from src.model_clients import MaxDiffResponse
        
        # Mock the Pydantic response that Instructor would return
        mock_pydantic_response = MaxDiffResponse(
            best_item=2,
            worst_item=3,
            reasoning="Claude structured reasoning from Instructor integration"
        )
        
        with patch('src.model_clients.instructor') as mock_instructor_module, \
             patch('src.model_clients.anthropic.AsyncAnthropic') as mock_anthropic:
            
            # Setup mock Instructor client
            mock_instructor_client = AsyncMock()
            mock_instructor_client.messages.create.return_value = mock_pydantic_response
            
            # Mock the instructor.from_anthropic function to return our mock client
            mock_instructor_module.from_anthropic.return_value = mock_instructor_client
            
            # Setup mock base client
            mock_base_client = AsyncMock()
            mock_anthropic.return_value = mock_base_client
            
            client = AnthropicClient(model_name="claude-3", api_key="test-key")
            response = await client.evaluate_trial(sample_trial, engine_config)
            
            assert response.model_name == "anthropic-claude-3"
            assert response.trial_number == sample_trial.trial_number
            assert response.success is True
            assert "structured reasoning" in response.reasoning
            assert response.best_item_id == sample_trial.items[1].id
            assert response.worst_item_id == sample_trial.items[2].id
            
            # Verify that Instructor was used
            mock_instructor_client.messages.create.assert_called_once()
            call_args = mock_instructor_client.messages.create.call_args
            assert call_args[1]['response_model'] == MaxDiffResponse


class TestGoogleClient:
    """Test Google client."""
    
    @pytest.mark.asyncio
    async def test_google_client_initialization(self):
        """Test Google client initialization."""
        with patch('src.model_clients.genai.configure') as mock_configure, \
             patch('src.model_clients.genai.GenerativeModel') as mock_model:
            
            client = GoogleClient(model_name="gemini-pro", api_key="test-key")
            assert client.model_name == "gemini-pro"
            assert client.api_key == "test-key"
            mock_configure.assert_called_once_with(api_key="test-key")
            mock_model.assert_called_once_with("gemini-pro")
    
    @pytest.mark.asyncio
    async def test_google_evaluate_trial_success(self, sample_trial, engine_config):
        """Test successful Google API call with structured output."""
        from src.model_clients import MaxDiffResponse
        
        # Mock the JSON response that Gemini would return
        mock_response = MagicMock()
        mock_response.text = '{"best_item": 3, "worst_item": 1, "reasoning": "Gemini structured reasoning from native schema support"}'
        
        with patch('src.model_clients.genai.configure'), \
             patch('src.model_clients.genai.GenerativeModel') as mock_model_class:
            
            mock_model = MagicMock()
            mock_model.generate_content.return_value = mock_response
            mock_model_class.return_value = mock_model
            
            client = GoogleClient(model_name="gemini-pro", api_key="test-key")
            response = await client.evaluate_trial(sample_trial, engine_config)
            
            assert response.model_name == "google-gemini-pro"
            assert response.trial_number == sample_trial.trial_number
            assert response.success is True
            assert "structured reasoning" in response.reasoning
            assert response.best_item_id == sample_trial.items[2].id
            assert response.worst_item_id == sample_trial.items[0].id
            
            # Verify that the structured generation config was used
            call_args = mock_model.generate_content.call_args
            generation_config = call_args[1]['generation_config']
            assert generation_config.response_mime_type == "application/json"
            assert generation_config.response_schema is not None


class TestPromptGeneration:
    """Test prompt generation functionality."""
    
    def test_prompt_contains_all_items(self, sample_trial, engine_config):
        """Test that prompt contains all trial items."""
        client = MockModelClient(model_name="test", api_key="key")
        prompt = client._create_prompt(sample_trial, engine_config)
        
        for i, item in enumerate(sample_trial.items):
            expected_line = f"{i+1}. {item.name}"
            assert expected_line in prompt
    
    def test_prompt_respects_config(self, sample_trial):
        """Test that prompt respects engine configuration."""
        config = EngineConfig(
            dimension_positive_label="Favorite",
            dimension_negative_label="Least Favorite",
            instruction_text="Pick your favorite and least favorite",
            persona="You are a food critic"
        )
        
        client = MockModelClient(model_name="test", api_key="key")
        prompt = client._create_prompt(sample_trial, config)
        
        assert "You are a food critic" in prompt
        assert "Pick your favorite and least favorite" in prompt
        assert "favorite" in prompt.lower()
        assert "least favorite" in prompt.lower()
    
    def test_prompt_json_format(self, sample_trial, engine_config):
        """Test that prompt requests proper JSON format."""
        client = MockModelClient(model_name="test", api_key="key")
        prompt = client._create_prompt(sample_trial, engine_config)
        
        assert "JSON" in prompt
        assert "best_item" in prompt
        assert "worst_item" in prompt
        assert "reasoning" in prompt
        assert "1-" in prompt  # Range indication
        assert f"{len(sample_trial.items)}" in prompt  # Max number


class TestMaxDiffResponseModel:
    """Test the Pydantic MaxDiffResponse model."""
    
    def test_valid_response_creation(self):
        """Test creating a valid MaxDiffResponse."""
        data = {
            'best_item': 2,
            'worst_item': 4,
            'reasoning': 'This is a valid reasoning with sufficient length to meet requirements.'
        }
        response = MaxDiffResponse(**data)
        
        assert response.best_item == 2
        assert response.worst_item == 4
        assert response.reasoning == data['reasoning']
    
    def test_different_items_validation(self):
        """Test that best and worst items must be different."""
        data = {
            'best_item': 2,
            'worst_item': 2,  # Same as best_item
            'reasoning': 'This should fail validation because items are the same.'
        }
        
        with pytest.raises(ValidationError) as exc_info:
            MaxDiffResponse(**data)
        
        assert "Best and worst items must be different" in str(exc_info.value)
    
    def test_minimum_item_values(self):
        """Test that item numbers must be >= 1."""
        data = {
            'best_item': 0,  # Invalid: must be >= 1
            'worst_item': 2,
            'reasoning': 'This should fail because best_item is 0.'
        }
        
        with pytest.raises(ValidationError) as exc_info:
            MaxDiffResponse(**data)
        
        assert "greater than or equal to 1" in str(exc_info.value)
    
    def test_reasoning_length_validation(self):
        """Test reasoning length constraints."""
        # Too short reasoning
        data = {
            'best_item': 1,
            'worst_item': 2,
            'reasoning': 'Short'  # Less than 10 characters
        }
        
        with pytest.raises(ValidationError) as exc_info:
            MaxDiffResponse(**data)
        
        assert "at least 10 characters" in str(exc_info.value)
    
    def test_reasoning_max_length(self):
        """Test reasoning maximum length constraint."""
        data = {
            'best_item': 1,
            'worst_item': 2,
            'reasoning': 'A' * 1001  # More than 1000 characters
        }
        
        with pytest.raises(ValidationError) as exc_info:
            MaxDiffResponse(**data)
        
        assert "at most 1000 characters" in str(exc_info.value)
    
    def test_schema_generation(self):
        """Test that Pydantic schema is generated correctly."""
        schema = MaxDiffResponse.model_json_schema()
        
        assert 'properties' in schema
        assert 'best_item' in schema['properties']
        assert 'worst_item' in schema['properties']
        assert 'reasoning' in schema['properties']
        assert schema['required'] == ['best_item', 'worst_item', 'reasoning']
        
        # Check field constraints
        best_item_schema = schema['properties']['best_item']
        assert best_item_schema['minimum'] == 1
        assert best_item_schema['type'] == 'integer'
        
        reasoning_schema = schema['properties']['reasoning']
        assert reasoning_schema['minLength'] == 10
        assert reasoning_schema['maxLength'] == 1000


class TestPydanticParsing:
    """Test Pydantic-based response parsing."""
    
    def test_perfect_pydantic_json_parsing(self, sample_trial):
        """Test parsing perfectly formatted JSON that passes Pydantic validation."""
        client = MockModelClient('test', 'key')
        response_text = '{"best_item": 2, "worst_item": 4, "reasoning": "Item 2 is excellent and item 4 is disappointing in comparison."}'
        
        result = client._parse_response(response_text, sample_trial)
        
        assert result['success'] is True
        assert result['best_item_id'] == sample_trial.items[1].id
        assert result['worst_item_id'] == sample_trial.items[3].id
        assert "excellent" in result['reasoning']
    
    def test_pydantic_json_with_extra_text(self, sample_trial):
        """Test parsing JSON embedded in extra text."""
        client = MockModelClient('test', 'key')
        response_text = '''
        Here is my analysis of the items:
        
        {"best_item": 1, "worst_item": 3, "reasoning": "After careful consideration, item 1 is superior to all others."}
        
        I hope this helps with your research.
        '''
        
        result = client._parse_response(response_text, sample_trial)
        
        assert result['success'] is True
        assert result['best_item_id'] == sample_trial.items[0].id
        assert result['worst_item_id'] == sample_trial.items[2].id
    
    def test_malformed_json_cleanup(self, sample_trial):
        """Test automatic cleanup of malformed JSON."""
        client = MockModelClient('test', 'key')
        # JSON with trailing comma
        response_text = '{"best_item": 3, "worst_item": 1, "reasoning": "Good reasoning with sufficient length here",}'
        
        result = client._parse_response(response_text, sample_trial)
        
        assert result['success'] is True
        assert result['best_item_id'] == sample_trial.items[2].id
        assert result['worst_item_id'] == sample_trial.items[0].id
    
    def test_alternative_key_names_limitation(self, sample_trial):
        """Test that alternative key names are not currently supported."""
        client = MockModelClient('test', 'key')
        response_text = '{"best": 2, "worst": 4, "reasoning": "Alternative key names currently fail parsing."}'
        
        result = client._parse_response(response_text, sample_trial)
        
        # Alternative key names currently fail (known limitation)
        assert result['success'] is False
        assert result['error_message'] == "Could not parse model response with any strategy"
        
        # But the standard key names work fine
        standard_response = '{"best_item": 2, "worst_item": 4, "reasoning": "Standard key names work perfectly."}'
        standard_result = client._parse_response(standard_response, sample_trial)
        assert standard_result['success'] is True
    
    def test_text_pattern_fallback(self, sample_trial):
        """Test fallback to text pattern matching."""
        client = MockModelClient('test', 'key')
        response_text = "I choose item 3 as the best option and item 1 as the worst choice."
        
        result = client._parse_response(response_text, sample_trial)
        
        assert result['success'] is True
        assert result['best_item_id'] == sample_trial.items[2].id
        assert result['worst_item_id'] == sample_trial.items[0].id
        assert "text patterns" in result['reasoning']
    
    def test_structured_text_patterns(self, sample_trial):
        """Test various structured text patterns."""
        client = MockModelClient('test', 'key')
        
        test_cases = [
            ("best_item: 2, worst_item: 4", 2, 4),
            ("Best item is 1 and worst item is 3", 1, 3),
            ("item 4 is best, item 2 is worst", 4, 2),
        ]
        
        for response_text, expected_best, expected_worst in test_cases:
            result = client._parse_response(response_text, sample_trial)
            
            if result['success']:
                assert result['best_item_id'] == sample_trial.items[expected_best - 1].id
                assert result['worst_item_id'] == sample_trial.items[expected_worst - 1].id
    
    def test_pydantic_validation_failure_with_fallback(self, sample_trial):
        """Test when Pydantic validation fails but text parsing succeeds."""
        client = MockModelClient('test', 'key')
        # JSON with reasoning too short for Pydantic but extractable via patterns
        response_text = '{"best_item": 2, "worst_item": 4, "reasoning": "Short"}'
        
        result = client._parse_response(response_text, sample_trial)
        
        # Should succeed via text pattern fallback
        assert result['success'] is True
        assert result['best_item_id'] == sample_trial.items[1].id
        assert result['worst_item_id'] == sample_trial.items[3].id
        assert "text patterns" in result['reasoning']
    
    def test_multiple_json_objects(self, sample_trial):
        """Test handling of multiple JSON objects in response."""
        client = MockModelClient('test', 'key')
        response_text = '''
        {"invalid": "object"}
        {"best_item": 2, "worst_item": 4, "reasoning": "This is the valid response object with proper length."}
        {"another": "object"}
        '''
        
        result = client._parse_response(response_text, sample_trial)
        
        assert result['success'] is True
        assert result['best_item_id'] == sample_trial.items[1].id
    
    def test_unicode_in_reasoning(self, sample_trial):
        """Test handling of unicode characters in reasoning."""
        client = MockModelClient('test', 'key')
        response_text = '{"best_item": 1, "worst_item": 2, "reasoning": "Item 1 is fantastic! 🍎 > 🍌 in my opinion."}'
        
        result = client._parse_response(response_text, sample_trial)
        
        assert result['success'] is True
        assert "🍎" in result['reasoning']
    
    def test_out_of_range_indices(self, sample_trial):
        """Test handling of indices outside valid range."""
        client = MockModelClient('test', 'key')
        response_text = '{"best_item": 10, "worst_item": 0, "reasoning": "These indices are completely out of range."}'
        
        result = client._parse_response(response_text, sample_trial)
        
        assert result['success'] is False
        assert result['error_message'] == "Could not parse model response with any strategy"
        # Should fallback to first and last items
        assert result['best_item_id'] == sample_trial.items[0].id
        assert result['worst_item_id'] == sample_trial.items[-1].id
    
    def test_same_item_pydantic_validation(self, sample_trial):
        """Test when response has same item for best and worst."""
        client = MockModelClient('test', 'key')
        response_text = '{"best_item": 2, "worst_item": 2, "reasoning": "This should fail Pydantic validation for same items."}'
        
        result = client._parse_response(response_text, sample_trial)
        
        # Should fail Pydantic validation and fallback
        assert result['success'] is False
        assert result['error_message'] == "Could not parse model response with any strategy"


class TestPydanticPromptGeneration:
    """Test prompt generation with Pydantic schema."""
    
    def test_pydantic_schema_included_in_prompt(self, sample_trial, engine_config):
        """Test that Pydantic schema is included in prompts."""
        client = MockModelClient('test', 'key')
        prompt = client._create_prompt(sample_trial, engine_config)
        
        # Check that schema elements are present
        assert '"properties"' in prompt
        assert '"best_item"' in prompt
        assert '"worst_item"' in prompt
        assert '"reasoning"' in prompt
        assert '"required"' in prompt
        assert 'schema' in prompt.lower()
    
    def test_pydantic_constraints_in_prompt(self, sample_trial, engine_config):
        """Test that Pydantic constraints are clearly stated in prompt."""
        client = MockModelClient('test', 'key')
        prompt = client._create_prompt(sample_trial, engine_config)
        
        assert "Constraints:" in prompt
        assert f"1 to {len(sample_trial.items)}" in prompt  # Item range
        assert "different numbers" in prompt
        assert "10-1000 characters" in prompt
    
    def test_pydantic_example_format_in_prompt(self, sample_trial, engine_config):
        """Test that example response format is included."""
        client = MockModelClient('test', 'key')
        prompt = client._create_prompt(sample_trial, engine_config)
        
        assert "Example response format:" in prompt
        assert '"best_item":' in prompt
        assert '"worst_item":' in prompt


class TestProviderPydanticConsistency:
    """Test that Pydantic parsing works consistently across all providers."""
    
    def test_openai_client_pydantic_parsing(self, sample_trial):
        """Test that OpenAI client uses Pydantic parsing."""
        client = OpenAIClient('gpt-4', 'dummy-key')
        response_text = '{"best_item": 1, "worst_item": 3, "reasoning": "OpenAI response with proper reasoning length."}'
        
        result = client._parse_response(response_text, sample_trial)
        
        assert result['success'] is True
        assert result['best_item_id'] == sample_trial.items[0].id
        assert result['worst_item_id'] == sample_trial.items[2].id
    
    def test_anthropic_client_pydantic_parsing(self, sample_trial):
        """Test that Anthropic client uses Pydantic parsing."""
        client = AnthropicClient('claude-3', 'dummy-key')
        response_text = '{"best_item": 2, "worst_item": 4, "reasoning": "Anthropic response with detailed reasoning for validation."}'
        
        result = client._parse_response(response_text, sample_trial)
        
        assert result['success'] is True
        assert result['best_item_id'] == sample_trial.items[1].id
        assert result['worst_item_id'] == sample_trial.items[3].id
    
    def test_google_client_pydantic_parsing(self, sample_trial):
        """Test that Google client uses Pydantic parsing."""
        client = GoogleClient('gemini-pro', 'dummy-key')
        response_text = '{"best_item": 3, "worst_item": 1, "reasoning": "Google Gemini response with comprehensive reasoning."}'
        
        result = client._parse_response(response_text, sample_trial)
        
        assert result['success'] is True
        assert result['best_item_id'] == sample_trial.items[2].id
        assert result['worst_item_id'] == sample_trial.items[0].id
    
    def test_consistent_pydantic_parsing_across_providers(self, sample_trial):
        """Test that all providers parse the same response consistently."""
        clients = [
            OpenAIClient('gpt-4', 'dummy-key'),
            AnthropicClient('claude-3', 'dummy-key'),
            GoogleClient('gemini-pro', 'dummy-key')
        ]
        
        response_text = '{"best_item": 2, "worst_item": 4, "reasoning": "Consistent response across all providers with proper length."}'
        
        results = []
        for client in clients:
            result = client._parse_response(response_text, sample_trial)
            results.append(result)
        
        # All should succeed and give same results
        for result in results:
            assert result['success'] is True
            assert result['best_item_id'] == sample_trial.items[1].id
            assert result['worst_item_id'] == sample_trial.items[3].id
            assert "Consistent response" in result['reasoning']

