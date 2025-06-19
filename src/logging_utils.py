"""
Logging utilities for MaxDiff AI agents runs.
Provides CSV logging for run outputs and environment settings.
"""
import csv
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    from .types import TaskSession, ReportConfig, AggregatedResults
except ImportError:
    from src.types import TaskSession, ReportConfig, AggregatedResults


class MaxDiffLogger:
    """
    Comprehensive logging system for MaxDiff runs.
    Creates timestamped CSV files for run outputs and environment settings.
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize logger with data directory."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define CSV file paths with timestamps
        self.runs_csv = self.data_dir / f"maxdiff_runs_{self.timestamp}.csv"
        self.settings_csv = self.data_dir / f"maxdiff_settings_{self.timestamp}.csv"
        
        # Sensitive keys that should never be logged
        self.sensitive_keys = {
            'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY',
            'API_KEY', 'SECRET', 'TOKEN', 'PASSWORD', 'PRIVATE_KEY'
        }
    
    def log_run_results(self, session: TaskSession, results: AggregatedResults, 
                       report_config: ReportConfig, execution_time: float = None) -> None:
        """
        Log comprehensive run results to CSV.
        
        Args:
            session: The task session with all trial data
            results: Aggregated results from the run
            report_config: Report configuration used
            execution_time: Total execution time in seconds
        """
        # Create item name mapping
        item_names = {item.id: item.name for item in session.items}
        
        # Prepare main run data
        run_data = {
            'timestamp': datetime.now().isoformat(),
            'run_id': self.timestamp,
            'total_trials': results.total_trials,
            'total_items': len(session.items),
            'models_used': ', '.join(results.models_used),
            'model_count': len(results.models_used),
            'successful_responses': len([r for r in session.responses if r.success]),
            'failed_responses': len([r for r in session.responses if not r.success]),
            'success_rate': len([r for r in session.responses if r.success]) / len(session.responses) if session.responses else 0,
            'items_per_subset': session.config.items_per_subset,
            'target_trials': session.config.target_trials,
            'dimension_positive_label': session.config.dimension_positive_label,
            'dimension_negative_label': session.config.dimension_negative_label,
            'persona': session.config.persona,
            'disagreement_points_count': len(results.disagreement_points),
            'execution_time_seconds': execution_time,
            'output_format': report_config.output_format,
            'report_style': report_config.report_style,
            'include_raw_responses': report_config.include_raw_responses
        }
        
        # Add top 3 consensus items
        for i, item_id in enumerate(results.consensus_ranking[:3]):
            item_name = item_names.get(item_id, 'Unknown')
            utility_score = results.item_scores.get(item_id, {}).get('utility_score', 0)
            run_data[f'top_{i+1}_item'] = item_name
            run_data[f'top_{i+1}_utility_score'] = utility_score
        
        # Add bottom 3 consensus items
        for i, item_id in enumerate(results.consensus_ranking[-3:]):
            item_name = item_names.get(item_id, 'Unknown')
            utility_score = results.item_scores.get(item_id, {}).get('utility_score', 0)
            run_data[f'bottom_{i+1}_item'] = item_name
            run_data[f'bottom_{i+1}_utility_score'] = utility_score
        
        # Calculate agreement metrics
        if results.agreement_matrix:
            std_devs = [item.get('utility_std_dev', 0) for item in results.agreement_matrix]
            run_data['mean_utility_std_dev'] = sum(std_devs) / len(std_devs) if std_devs else 0
            run_data['max_utility_std_dev'] = max(std_devs) if std_devs else 0
            run_data['min_utility_std_dev'] = min(std_devs) if std_devs else 0
        
        # Write to CSV
        self._write_csv_row(self.runs_csv, run_data)
        
        # Also log individual item results
        self._log_item_results(session, results)
        
        # Log individual responses
        self._log_response_details(session)
    
    def log_environment_settings(self, env_vars: Dict[str, Any]) -> None:
        """
        Log environment settings to CSV, excluding sensitive information.
        
        Args:
            env_vars: Dictionary of environment variables and settings
        """
        # Filter out sensitive keys
        safe_env_vars = {}
        for key, value in env_vars.items():
            if any(sensitive in key.upper() for sensitive in self.sensitive_keys):
                safe_env_vars[key] = '[REDACTED]'
            else:
                safe_env_vars[key] = str(value) if value is not None else ''
        
        settings_data = {
            'timestamp': datetime.now().isoformat(),
            'run_id': self.timestamp,
            **safe_env_vars
        }
        
        self._write_csv_row(self.settings_csv, settings_data)
    
    def _log_item_results(self, session: TaskSession, results: AggregatedResults) -> None:
        """Log detailed item-level results to a separate CSV."""
        item_results_csv = self.data_dir / f"maxdiff_item_results_{self.timestamp}.csv"
        item_names = {item.id: item.name for item in session.items}
        
        # Create agreement data mapping
        agreement_data_map = {item['item_id']: item for item in results.agreement_matrix}
        
        for i, item_id in enumerate(results.consensus_ranking):
            scores = results.item_scores.get(item_id, {})
            agreement_data = agreement_data_map.get(item_id, {})
            
            item_data = {
                'timestamp': datetime.now().isoformat(),
                'run_id': self.timestamp,
                'item_id': item_id,
                'item_name': item_names.get(item_id, 'Unknown'),
                'consensus_rank': i + 1,
                'utility_score': scores.get('utility_score', 0),
                'best_count': scores.get('best_count', 0),
                'worst_count': scores.get('worst_count', 0),
                'appearance_count': scores.get('appearance_count', 0),
                'best_rate': scores.get('best_rate', 0),
                'worst_rate': scores.get('worst_rate', 0),
                'net_score': scores.get('net_score', 0),
                'utility_std_dev': agreement_data.get('utility_std_dev', 0),
                'mean_utility': agreement_data.get('mean_utility', 0),
                'agreement_score': agreement_data.get('agreement_score', 0),
                'is_disagreement_point': item_id in [d['item_id'] for d in results.disagreement_points]
            }
            
            # Add model-specific scores if available
            if 'model_scores' in agreement_data:
                for model_score in agreement_data['model_scores']:
                    model_name = model_score['model'].replace('-', '_').replace('.', '_')
                    item_data[f'{model_name}_utility_score'] = model_score.get('utility_score', 0)
                    item_data[f'{model_name}_best_rate'] = model_score.get('best_rate', 0)
                    item_data[f'{model_name}_worst_rate'] = model_score.get('worst_rate', 0)
                    item_data[f'{model_name}_appearances'] = model_score.get('appearances', 0)
            
            self._write_csv_row(item_results_csv, item_data)
    
    def _log_response_details(self, session: TaskSession) -> None:
        """Log individual response details to CSV."""
        responses_csv = self.data_dir / f"maxdiff_responses_{self.timestamp}.csv"
        item_names = {item.id: item.name for item in session.items}
        
        for response in session.responses:
            # Find the corresponding trial
            trial = next((t for t in session.trials if t.trial_number == response.trial_number), None)
            trial_items = [item_names.get(item.id, 'Unknown') for item in trial.items] if trial else []
            
            response_data = {
                'timestamp': datetime.now().isoformat(),
                'run_id': self.timestamp,
                'trial_number': response.trial_number,
                'model_name': response.model_name,
                'success': response.success,
                'best_item_id': response.best_item_id,
                'worst_item_id': response.worst_item_id,
                'best_item_name': item_names.get(response.best_item_id, 'Unknown') if response.best_item_id else '',
                'worst_item_name': item_names.get(response.worst_item_id, 'Unknown') if response.worst_item_id else '',
                'trial_items': ', '.join(trial_items),
                'trial_items_count': len(trial_items),
                'reasoning_length': len(response.reasoning) if response.reasoning else 0,
                'has_reasoning': bool(response.reasoning),
                'error_message': response.error_message or '',
                'has_error': bool(response.error_message)
            }
            
            self._write_csv_row(responses_csv, response_data)
    
    def _write_csv_row(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Write a single row to CSV file, creating headers if file doesn't exist."""
        file_exists = file_path.exists()
        
        with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
            if data:
                writer = csv.DictWriter(csvfile, fieldnames=data.keys())
                
                # Write header if file is new
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(data)
    
    def get_run_summary(self) -> Dict[str, str]:
        """Get summary of files created for this run."""
        return {
            'run_id': self.timestamp,
            'runs_csv': str(self.runs_csv),
            'settings_csv': str(self.settings_csv),
            'item_results_csv': str(self.data_dir / f"maxdiff_item_results_{self.timestamp}.csv"),
            'responses_csv': str(self.data_dir / f"maxdiff_responses_{self.timestamp}.csv")
        }


def get_environment_settings() -> Dict[str, Any]:
    """
    Collect all relevant environment settings for logging.
    
    Returns:
        Dictionary of environment variables and computed settings
    """
    return {
        # MaxDiff Configuration
        'items_per_subset': os.getenv('ITEMS_PER_SUBSET', '4'),
        'target_trials': os.getenv('TARGET_TRIALS', '20'),
        'dimension_positive_label': os.getenv('DIMENSION_POSITIVE_LABEL', 'Best'),
        'dimension_negative_label': os.getenv('DIMENSION_NEGATIVE_LABEL', 'Worst'),
        'instruction_text': os.getenv('INSTRUCTION_TEXT', 'Please choose the item you find BEST and the item you find WORST.'),
        'persona': os.getenv('PERSONA', 'You are an expert evaluating these items objectively'),
        
        # Model Configuration
        'openai_model': os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
        'anthropic_model': os.getenv('ANTHROPIC_MODEL', 'claude-3-haiku-20240307'),
        'google_model': os.getenv('GOOGLE_MODEL', 'gemini-1.5-flash'),
        'has_openai_key': bool(os.getenv('OPENAI_API_KEY')),
        'has_anthropic_key': bool(os.getenv('ANTHROPIC_API_KEY')),
        'has_google_key': bool(os.getenv('GOOGLE_API_KEY')),
        
        # Performance & Output
        'max_concurrent_requests': os.getenv('MAX_CONCURRENT_REQUESTS', '5'),
        'request_timeout': os.getenv('REQUEST_TIMEOUT', '30'),
        'output_format': os.getenv('OUTPUT_FORMAT', 'html'),
        'include_raw_responses': os.getenv('INCLUDE_RAW_RESPONSES', 'false'),
        'report_style': os.getenv('REPORT_STYLE', 'detailed'),
        'report_output_file': os.getenv('REPORT_OUTPUT_FILE', 'report.html'),
        
        # System Information
        'python_version': os.sys.version.split()[0],
        'platform': os.name,
        'working_directory': os.getcwd()
    }

