"""
Types and data models for the MaxDiff AI agents project.
"""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from dataclasses import dataclass
import uuid


class MaxDiffItem(BaseModel):
    """Represents a single item to be evaluated in MaxDiff tasks."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: Literal['text'] = 'text'
    text_content: Optional[str] = None
    
    def __post_init__(self):
        if self.text_content is None:
            self.text_content = self.name


class EngineConfig(BaseModel):
    """Configuration for the MaxDiff engine."""
    items_per_subset: int = Field(default=4, ge=2, le=10)
    target_trials: int = Field(default=20, ge=1)
    dimension_positive_label: str = "Best"
    dimension_negative_label: str = "Worst"
    instruction_text: str = "Please choose the item you find BEST and the item you find WORST."


class RecordedChoice(BaseModel):
    """Records a choice made in a MaxDiff trial."""
    trial_number: int
    presented_item_ids: List[str]
    best_item_id: str
    worst_item_id: str
    model_name: str
    reasoning: Optional[str] = None
    timestamp: Optional[str] = None


class TrialSet(BaseModel):
    """Represents a set of items for a single MaxDiff trial."""
    trial_number: int
    items: List[MaxDiffItem]
    
    
class ModelResponse(BaseModel):
    """Response from an AI model for a MaxDiff trial."""
    model_name: str
    trial_number: int
    best_item_id: str
    worst_item_id: str
    reasoning: Optional[str] = None
    raw_response: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


class ModelConfig(BaseModel):
    """Configuration for AI model clients."""
    openai_model: str = "gpt-3.5-turbo"
    anthropic_model: str = "claude-3-haiku-20240307"
    google_model: str = "gemini-1.5-flash"
    max_concurrent_requests: int = 5
    request_timeout: int = 30


class AggregatedResults(BaseModel):
    """Aggregated results from all models and trials."""
    total_trials: int
    models_used: List[str]
    item_scores: Dict[str, Dict[str, float]]  # item_id -> {metric: score}
    consensus_ranking: List[str]  # item_ids in order of preference
    agreement_matrix: Dict[str, Dict[str, float]]  # model1 -> model2 -> agreement_score
    disagreement_points: List[Dict[str, Any]]
    
    
class ReportConfig(BaseModel):
    """Configuration for report generation."""
    output_format: Literal['json', 'html', 'markdown'] = 'json'
    include_raw_responses: bool = False
    report_style: Literal['summary', 'detailed'] = 'detailed'
    output_file: Optional[str] = None


@dataclass
class TaskSession:
    """Represents a complete MaxDiff task session."""
    items: List[MaxDiffItem]
    config: EngineConfig
    trials: List[TrialSet]
    responses: List[ModelResponse]
    results: Optional[AggregatedResults] = None

