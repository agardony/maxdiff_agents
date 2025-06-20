"""
Tests for the types module.
"""
import pytest
import uuid
from pydantic import ValidationError

from src.types import (
    MaxDiffItem, EngineConfig, RecordedChoice, TrialSet, 
    ModelResponse, ModelConfig, AggregatedResults, ReportConfig, TaskSession
)


class TestMaxDiffItem:
    """Test MaxDiffItem class."""
    
    def test_create_item_with_name(self):
        """Test creating an item with just a name."""
        item = MaxDiffItem(name="Apple")
        assert item.name == "Apple"
        assert item.type == "text"
        assert item.text_content is None
        assert isinstance(item.id, str)
        # Verify it's a valid UUID
        uuid.UUID(item.id)
    
    def test_create_item_with_all_fields(self):
        """Test creating an item with all fields."""
        item_id = str(uuid.uuid4())
        item = MaxDiffItem(
            id=item_id,
            name="Banana",
            type="text",
            text_content="A yellow fruit"
        )
        assert item.id == item_id
        assert item.name == "Banana"
        assert item.type == "text"
        assert item.text_content == "A yellow fruit"
    
    def test_unique_ids(self):
        """Test that different items get unique IDs."""
        item1 = MaxDiffItem(name="Apple")
        item2 = MaxDiffItem(name="Banana")
        assert item1.id != item2.id


class TestEngineConfig:
    """Test EngineConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EngineConfig()
        assert config.items_per_subset == 4
        assert config.target_trials == 20
        assert config.dimension_positive_label == "Best"
        assert config.dimension_negative_label == "Worst"
        assert "BEST" in config.instruction_text
        assert "WORST" in config.instruction_text
        assert "expert" in config.persona
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = EngineConfig(
            items_per_subset=3,
            target_trials=10,
            dimension_positive_label="Good",
            dimension_negative_label="Bad",
            instruction_text="Choose good and bad",
            persona="You are a critic"
        )
        assert config.items_per_subset == 3
        assert config.target_trials == 10
        assert config.dimension_positive_label == "Good"
        assert config.dimension_negative_label == "Bad"
        assert config.instruction_text == "Choose good and bad"
        assert config.persona == "You are a critic"
    
    def test_validation_constraints(self):
        """Test validation constraints."""
        # items_per_subset must be >= 2
        with pytest.raises(ValidationError):
            EngineConfig(items_per_subset=1)
        
        # items_per_subset must be <= 10
        with pytest.raises(ValidationError):
            EngineConfig(items_per_subset=11)
        
        # target_trials must be >= 1
        with pytest.raises(ValidationError):
            EngineConfig(target_trials=0)


class TestRecordedChoice:
    """Test RecordedChoice class."""
    
    def test_create_recorded_choice(self):
        """Test creating a recorded choice."""
        choice = RecordedChoice(
            trial_number=1,
            presented_item_ids=["id1", "id2", "id3", "id4"],
            best_item_id="id1",
            worst_item_id="id4",
            model_name="test-model",
            reasoning="Test reasoning"
        )
        assert choice.trial_number == 1
        assert len(choice.presented_item_ids) == 4
        assert choice.best_item_id == "id1"
        assert choice.worst_item_id == "id4"
        assert choice.model_name == "test-model"
        assert choice.reasoning == "Test reasoning"


class TestTrialSet:
    """Test TrialSet class."""
    
    def test_create_trial_set(self, sample_items):
        """Test creating a trial set."""
        trial = TrialSet(
            trial_number=1,
            items=sample_items[:4]
        )
        assert trial.trial_number == 1
        assert len(trial.items) == 4
        assert all(isinstance(item, MaxDiffItem) for item in trial.items)


class TestModelResponse:
    """Test ModelResponse class."""
    
    def test_successful_response(self):
        """Test creating a successful model response."""
        response = ModelResponse(
            model_name="test-model",
            trial_number=1,
            best_item_id="item1",
            worst_item_id="item2",
            reasoning="Test reasoning",
            raw_response='{"best": 1, "worst": 2}',
            success=True
        )
        assert response.model_name == "test-model"
        assert response.trial_number == 1
        assert response.best_item_id == "item1"
        assert response.worst_item_id == "item2"
        assert response.success is True
        assert response.error_message is None
    
    def test_failed_response(self):
        """Test creating a failed model response."""
        response = ModelResponse(
            model_name="test-model",
            trial_number=1,
            best_item_id="item1",
            worst_item_id="item2",
            success=False,
            error_message="API timeout"
        )
        assert response.success is False
        assert response.error_message == "API timeout"


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_default_model_config(self):
        """Test default model configuration."""
        config = ModelConfig()
        assert config.openai_model == "gpt-3.5-turbo"
        assert config.anthropic_model == "claude-3-haiku-20240307"
        assert config.google_model == "gemini-1.5-flash"
        assert config.max_concurrent_requests == 5
        assert config.request_timeout == 30
        assert config.max_retries == 3
        assert config.retry_base_delay == 1.0
        assert config.retry_max_delay == 60.0
    
    def test_custom_model_config(self):
        """Test custom model configuration."""
        config = ModelConfig(
            max_retries=5,
            retry_base_delay=2.0,
            retry_max_delay=120.0,
            request_timeout=60
        )
        assert config.max_retries == 5
        assert config.retry_base_delay == 2.0
        assert config.retry_max_delay == 120.0
        assert config.request_timeout == 60


class TestReportConfig:
    """Test ReportConfig class."""
    
    def test_default_report_config(self):
        """Test default report configuration."""
        config = ReportConfig()
        assert config.output_format == 'json'
        assert config.include_raw_responses is False
        assert config.report_style == 'detailed'
        assert config.output_file is None
    
    def test_custom_report_config(self):
        """Test custom report configuration."""
        config = ReportConfig(
            output_format='html',
            include_raw_responses=True,
            report_style='summary',
            output_file='test_report.html'
        )
        assert config.output_format == 'html'
        assert config.include_raw_responses is True
        assert config.report_style == 'summary'
        assert config.output_file == 'test_report.html'
    
    def test_invalid_output_format(self):
        """Test invalid output format raises validation error."""
        with pytest.raises(ValidationError):
            ReportConfig(output_format='pdf')
    
    def test_invalid_report_style(self):
        """Test invalid report style raises validation error."""
        with pytest.raises(ValidationError):
            ReportConfig(report_style='verbose')


class TestAggregatedResults:
    """Test AggregatedResults class."""
    
    def test_create_aggregated_results(self):
        """Test creating aggregated results."""
        results = AggregatedResults(
            total_trials=10,
            models_used=["model1", "model2"],
            item_scores={"item1": {"utility_score": 0.5}},
            consensus_ranking=["item1", "item2"],
            agreement_matrix=[{"item_id": "item1", "agreement_score": 0.8}],
            disagreement_points=[{"item_id": "item2", "std_dev": 0.3}]
        )
        assert results.total_trials == 10
        assert len(results.models_used) == 2
        assert "item1" in results.item_scores
        assert len(results.consensus_ranking) == 2
        assert len(results.agreement_matrix) == 1
        assert len(results.disagreement_points) == 1


class TestTaskSession:
    """Test TaskSession dataclass."""
    
    def test_create_task_session(self, sample_items, engine_config):
        """Test creating a task session."""
        trials = [TrialSet(trial_number=1, items=sample_items[:4])]
        responses = [ModelResponse(
            model_name="test",
            trial_number=1, 
            best_item_id=sample_items[0].id,
            worst_item_id=sample_items[1].id
        )]
        
        session = TaskSession(
            items=sample_items,
            config=engine_config,
            trials=trials,
            responses=responses
        )
        
        assert len(session.items) == 6
        assert isinstance(session.config, EngineConfig)
        assert len(session.trials) == 1
        assert len(session.responses) == 1
        assert session.results is None

