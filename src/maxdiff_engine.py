"""
MaxDiff Engine implementation for generating trials and managing task sessions.
"""
import random
from typing import List, Optional
try:
    from .types import MaxDiffItem, EngineConfig, TrialSet, RecordedChoice
except ImportError:
    from src.types import MaxDiffItem, EngineConfig, TrialSet, RecordedChoice


class MaxDiffEngine:
    """
    Manages the logic for a MaxDiff task session, including
    trial generation, choice recording, and progress tracking.
    """
    
    def __init__(self, items: List[MaxDiffItem], config: EngineConfig):
        """
        Creates an instance of MaxDiffEngine.
        
        Args:
            items: The list of all items for the MaxDiff task.
            config: Configuration for the engine.
            
        Raises:
            ValueError: If the number of items is less than items_per_subset.
        """
        if len(items) < config.items_per_subset:
            raise ValueError('Number of items cannot be less than items_per_subset.')
        if config.items_per_subset < 2:
            raise ValueError('Items per subset must be at least 2 to pick best and worst.')
        
        # Store a copy to prevent external modification of the items list during a session
        self.items = items.copy()
        self.config = config
        self.trials_conducted = 0
        self.choices: List[RecordedChoice] = []
        self._generated_trials: List[TrialSet] = []
    
    def generate_all_trials(self) -> List[TrialSet]:
        """
        Generates all trial sets for the MaxDiff task.
        
        Returns:
            List of TrialSet objects containing items for each trial.
        """
        trials = []
        for trial_num in range(1, self.config.target_trials + 1):
            # Simple random sampling: shuffle a copy of items and take the first N
            shuffled_items = self.items.copy()
            random.shuffle(shuffled_items)
            trial_items = shuffled_items[:self.config.items_per_subset]
            
            trial_set = TrialSet(
                trial_number=trial_num,
                items=trial_items
            )
            trials.append(trial_set)
        
        self._generated_trials = trials
        return trials
    
    def get_next_trial_set(self) -> Optional[TrialSet]:
        """
        Generates the next set of items for a trial.
        
        Returns:
            TrialSet for the next trial, or None if the task is complete.
        """
        if self.is_complete():
            return None
        
        trial_num = self.trials_conducted + 1
        
        # Simple random sampling: shuffle a copy of items and take the first N
        shuffled_items = self.items.copy()
        random.shuffle(shuffled_items)
        trial_items = shuffled_items[:self.config.items_per_subset]
        
        return TrialSet(
            trial_number=trial_num,
            items=trial_items
        )
    
    def record_choice(
        self,
        presented_items: List[MaxDiffItem],
        best_item: MaxDiffItem,
        worst_item: MaxDiffItem,
        model_name: str,
        reasoning: Optional[str] = None
    ) -> None:
        """
        Records a choice for a given trial.
        
        Args:
            presented_items: The array of items presented in the trial.
            best_item: The item chosen as best.
            worst_item: The item chosen as worst.
            model_name: Name of the model making the choice.
            reasoning: Optional reasoning for the choice.
            
        Raises:
            ValueError: If bestItem and worstItem are the same, or not part of presentedItems.
        """
        
        if best_item.id == worst_item.id:
            raise ValueError('Best and worst items cannot be the same.')
        
        presented_ids = {item.id for item in presented_items}
        if best_item.id not in presented_ids or worst_item.id not in presented_ids:
            raise ValueError('Best and/or worst items are not part of the presented items.')
        
        self.trials_conducted += 1
        choice = RecordedChoice(
            trial_number=self.trials_conducted,
            presented_item_ids=[item.id for item in presented_items],
            best_item_id=best_item.id,
            worst_item_id=worst_item.id,
            model_name=model_name,
            reasoning=reasoning
        )
        self.choices.append(choice)
    
    def is_complete(self) -> bool:
        """
        Checks if the MaxDiff task is complete.
        
        Returns:
            True if complete, false otherwise.
        """
        return self.trials_conducted >= self.config.target_trials
    
    def get_choices(self) -> List[RecordedChoice]:
        """
        Retrieves all recorded choices.
        
        Returns:
            List of recorded choices.
        """
        return self.choices.copy()
    
    def get_progress(self) -> dict:
        """
        Gets the current progress of the task.
        
        Returns:
            Dict with conducted and target trials.
        """
        return {
            'conducted': self.trials_conducted,
            'target': self.config.target_trials
        }
    
    def get_generated_trials(self) -> List[TrialSet]:
        """
        Returns the generated trials for the session.
        
        Returns:
            List of generated TrialSet objects.
        """
        return self._generated_trials.copy()

