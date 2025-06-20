"""
Pytest configuration and shared fixtures for MaxDiff tests.
"""
import pytest
import sys
import os
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.types import MaxDiffItem, EngineConfig, TrialSet, ModelResponse, TaskSession, ReportConfig, ModelConfig, AggregatedResults
from src.maxdiff_engine import MaxDiffEngine


@pytest.fixture
def sample_items():
    """Create sample items for testing."""
    return [
        MaxDiffItem(name="Apple"),
        MaxDiffItem(name="Banana"), 
        MaxDiffItem(name="Cherry"),
        MaxDiffItem(name="Date"),
        MaxDiffItem(name="Elderberry"),
        MaxDiffItem(name="Fig")
    ]


@pytest.fixture
def engine_config():
    """Create a basic engine configuration."""
    return EngineConfig(
        items_per_subset=4,
        target_trials=5,
        dimension_positive_label="Best",
        dimension_negative_label="Worst",
        instruction_text="Please choose the item you find BEST and the item you find WORST.",
        persona="You are an expert evaluating these items objectively"
    )


@pytest.fixture
def model_config():
    """Create a basic model configuration."""
    return ModelConfig(
        max_retries=2,
        retry_base_delay=0.1,
        retry_max_delay=1.0,
        request_timeout=5
    )


@pytest.fixture
def maxdiff_engine(sample_items, engine_config):
    """Create a MaxDiff engine with sample data."""
    return MaxDiffEngine(items=sample_items, config=engine_config)


@pytest.fixture
def sample_trial(sample_items):
    """Create a sample trial set."""
    return TrialSet(
        trial_number=1,
        items=sample_items[:4]  # First 4 items
    )


@pytest.fixture
def sample_responses(sample_items):
    """Create sample model responses."""
    return [
        ModelResponse(
            model_name="test-model-1",
            trial_number=1,
            best_item_id=sample_items[0].id,
            worst_item_id=sample_items[3].id,
            reasoning="Apple is fresh, Date is old",
            success=True
        ),
        ModelResponse(
            model_name="test-model-2", 
            trial_number=1,
            best_item_id=sample_items[1].id,
            worst_item_id=sample_items[2].id,
            reasoning="Banana is nutritious, Cherry is too sweet",
            success=True
        ),
        ModelResponse(
            model_name="test-model-1",
            trial_number=2,
            best_item_id=sample_items[2].id,
            worst_item_id=sample_items[1].id,
            reasoning="Cherry is delicious, Banana is bland",
            success=True
        )
    ]


@pytest.fixture
def sample_task_session(sample_items, engine_config, sample_responses):
    """Create a complete task session for testing."""
    # Create trials
    trials = [
        TrialSet(trial_number=1, items=sample_items[:4]),
        TrialSet(trial_number=2, items=sample_items[1:5])
    ]
    
    return TaskSession(
        items=sample_items,
        config=engine_config,
        trials=trials,
        responses=sample_responses
    )


@pytest.fixture
def report_config():
    """Create a basic report configuration."""
    return ReportConfig(
        output_format='json',
        include_raw_responses=False,
        report_style='detailed'
    )


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory for testing."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return str(data_dir)

