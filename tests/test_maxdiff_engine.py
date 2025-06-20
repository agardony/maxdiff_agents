"""
Tests for the MaxDiff engine module.
"""
import pytest
import random

from src.maxdiff_engine import MaxDiffEngine
from src.types import MaxDiffItem, EngineConfig, TrialSet, RecordedChoice


class TestMaxDiffEngine:
    """Test MaxDiffEngine class."""
    
    def test_engine_initialization(self, sample_items, engine_config):
        """Test engine initialization."""
        engine = MaxDiffEngine(items=sample_items, config=engine_config)
        
        assert len(engine.items) == 6
        assert engine.config == engine_config
        assert engine.trials_conducted == 0
        assert len(engine.choices) == 0
        assert len(engine._generated_trials) == 0
    
    def test_engine_initialization_too_few_items(self, engine_config):
        """Test engine initialization with too few items."""
        items = [MaxDiffItem(name="Apple"), MaxDiffItem(name="Banana")]
        
        with pytest.raises(ValueError, match="Number of items cannot be less than items_per_subset"):
            MaxDiffEngine(items=items, config=engine_config)
    
    def test_engine_initialization_invalid_subset_size(self, sample_items):
        """Test engine initialization with invalid subset size."""
        # Test with pydantic validation error for items_per_subset < 2
        with pytest.raises(ValueError, match="Items per subset must be at least 2"):
            # This should fail in the MaxDiffEngine constructor, not in pydantic validation
            # Let's create a config that passes pydantic but fails engine validation
            config = EngineConfig(items_per_subset=2)
            config.items_per_subset = 1  # Bypass pydantic validation
            MaxDiffEngine(items=sample_items, config=config)
    
    def test_generate_all_trials(self, maxdiff_engine):
        """Test generating all trials."""
        trials = maxdiff_engine.generate_all_trials()
        
        assert len(trials) == 5  # target_trials from fixture
        assert all(isinstance(trial, TrialSet) for trial in trials)
        assert all(len(trial.items) == 4 for trial in trials)  # items_per_subset
        assert all(trial.trial_number == i + 1 for i, trial in enumerate(trials))
        
        # Check that generated trials are stored
        assert len(maxdiff_engine._generated_trials) == 5
    
    def test_get_next_trial_set(self, maxdiff_engine):
        """Test getting next trial set."""
        # First trial
        trial1 = maxdiff_engine.get_next_trial_set()
        assert trial1 is not None
        assert trial1.trial_number == 1
        assert len(trial1.items) == 4
        
        # Simulate conducting a trial
        maxdiff_engine.trials_conducted = 1
        
        trial2 = maxdiff_engine.get_next_trial_set()
        assert trial2.trial_number == 2
        
        # Test completion
        maxdiff_engine.trials_conducted = 5  # target_trials
        trial_complete = maxdiff_engine.get_next_trial_set()
        assert trial_complete is None
    
    def test_record_choice_valid(self, maxdiff_engine, sample_items):
        """Test recording a valid choice."""
        presented_items = sample_items[:4]
        best_item = sample_items[0]
        worst_item = sample_items[3]
        
        maxdiff_engine.record_choice(
            presented_items=presented_items,
            best_item=best_item,
            worst_item=worst_item,
            model_name="test-model",
            reasoning="Test reasoning"
        )
        
        assert maxdiff_engine.trials_conducted == 1
        assert len(maxdiff_engine.choices) == 1
        
        choice = maxdiff_engine.choices[0]
        assert choice.trial_number == 1
        assert choice.best_item_id == best_item.id
        assert choice.worst_item_id == worst_item.id
        assert choice.model_name == "test-model"
        assert choice.reasoning == "Test reasoning"
        assert len(choice.presented_item_ids) == 4
    
    def test_record_choice_same_best_worst(self, maxdiff_engine, sample_items):
        """Test recording choice with same best and worst item."""
        presented_items = sample_items[:4]
        same_item = sample_items[0]
        
        with pytest.raises(ValueError, match="Best and worst items cannot be the same"):
            maxdiff_engine.record_choice(
                presented_items=presented_items,
                best_item=same_item,
                worst_item=same_item,
                model_name="test-model"
            )
    
    def test_record_choice_item_not_presented(self, maxdiff_engine, sample_items):
        """Test recording choice with item not in presented items."""
        presented_items = sample_items[:4]
        best_item = sample_items[0]  # Valid
        worst_item = sample_items[5]  # Not in presented items
        
        with pytest.raises(ValueError, match="Best and/or worst items are not part of the presented items"):
            maxdiff_engine.record_choice(
                presented_items=presented_items,
                best_item=best_item,
                worst_item=worst_item,
                model_name="test-model"
            )
    
    def test_is_complete(self, maxdiff_engine):
        """Test completion status."""
        assert not maxdiff_engine.is_complete()
        
        # Simulate conducting all trials
        maxdiff_engine.trials_conducted = 5  # target_trials
        assert maxdiff_engine.is_complete()
        
        # Test with more than target
        maxdiff_engine.trials_conducted = 6
        assert maxdiff_engine.is_complete()
    
    def test_get_choices(self, maxdiff_engine, sample_items):
        """Test getting recorded choices."""
        # Initially empty
        choices = maxdiff_engine.get_choices()
        assert len(choices) == 0
        
        # Record a choice
        presented_items = sample_items[:4]
        maxdiff_engine.record_choice(
            presented_items=presented_items,
            best_item=sample_items[0],
            worst_item=sample_items[3],
            model_name="test-model"
        )
        
        choices = maxdiff_engine.get_choices()
        assert len(choices) == 1
        assert isinstance(choices[0], RecordedChoice)
        
        # Verify it's a copy (modifications don't affect original)
        choices.append("fake_choice")
        assert len(maxdiff_engine.get_choices()) == 1
    
    def test_get_progress(self, maxdiff_engine):
        """Test getting progress information."""
        progress = maxdiff_engine.get_progress()
        assert progress['conducted'] == 0
        assert progress['target'] == 5
        
        # Simulate progress
        maxdiff_engine.trials_conducted = 3
        progress = maxdiff_engine.get_progress()
        assert progress['conducted'] == 3
        assert progress['target'] == 5
    
    def test_get_generated_trials(self, maxdiff_engine):
        """Test getting generated trials."""
        # Initially empty
        trials = maxdiff_engine.get_generated_trials()
        assert len(trials) == 0
        
        # Generate trials
        maxdiff_engine.generate_all_trials()
        trials = maxdiff_engine.get_generated_trials()
        assert len(trials) == 5
        assert all(isinstance(trial, TrialSet) for trial in trials)
        
        # Verify it's a copy
        trials.append("fake_trial")
        assert len(maxdiff_engine.get_generated_trials()) == 5
    
    def test_randomization_in_trials(self, sample_items, engine_config):
        """Test that trial generation includes randomization."""
        # Set a seed for reproducible testing
        random.seed(42)
        engine1 = MaxDiffEngine(items=sample_items, config=engine_config)
        trials1 = engine1.generate_all_trials()
        
        # Reset seed and create another engine
        random.seed(123)
        engine2 = MaxDiffEngine(items=sample_items, config=engine_config)
        trials2 = engine2.generate_all_trials()
        
        # The trials should be different (very high probability)
        trial1_items = [item.id for item in trials1[0].items]
        trial2_items = [item.id for item in trials2[0].items]
        
        # With different seeds, it's extremely unlikely the first trial is identical
        assert trial1_items != trial2_items or len(set(trial1_items) ^ set(trial2_items)) > 0
    
    def test_items_immutability(self, sample_items, engine_config):
        """Test that external modification of items doesn't affect engine."""
        original_count = len(sample_items)
        engine = MaxDiffEngine(items=sample_items, config=engine_config)
        
        # Modify the original list
        sample_items.append(MaxDiffItem(name="New Item"))
        
        # Engine should still have the original count
        assert len(engine.items) == original_count
        
        # Engine's items should be unaffected
        assert len(engine.items) == 6
        assert all(item.name != "New Item" for item in engine.items)
    
    def test_multiple_choices_tracking(self, maxdiff_engine, sample_items):
        """Test recording multiple choices."""
        # Record first choice
        maxdiff_engine.record_choice(
            presented_items=sample_items[:4],
            best_item=sample_items[0],
            worst_item=sample_items[3],
            model_name="model1"
        )
        
        # Record second choice
        maxdiff_engine.record_choice(
            presented_items=sample_items[1:5],
            best_item=sample_items[2],
            worst_item=sample_items[4],
            model_name="model2"
        )
        
        assert maxdiff_engine.trials_conducted == 2
        assert len(maxdiff_engine.choices) == 2
        
        choices = maxdiff_engine.get_choices()
        assert choices[0].trial_number == 1
        assert choices[1].trial_number == 2
        assert choices[0].model_name == "model1"
        assert choices[1].model_name == "model2"

