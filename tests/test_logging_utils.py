"""
Tests for the logging utilities module.
"""
import pytest
import os
import csv
from pathlib import Path
from unittest.mock import patch

from src.logging_utils import MaxDiffLogger, get_environment_settings
from src.types import TaskSession, ReportConfig, AggregatedResults
from src.reporting import aggregate_results


class TestMaxDiffLogger:
    """Test MaxDiffLogger class."""
    
    def test_logger_initialization(self, temp_data_dir):
        """Test logger initialization."""
        logger = MaxDiffLogger(data_dir=temp_data_dir)
        
        assert logger.data_dir == Path(temp_data_dir)
        assert logger.data_dir.exists()
        assert logger.run_dir.exists()
        assert logger.timestamp
        
        # Check that run directory follows expected naming pattern
        assert logger.run_dir.name.startswith("run_")
        assert logger.timestamp in logger.run_dir.name
        
        # Check CSV file paths
        assert logger.runs_csv.parent == logger.run_dir
        assert logger.settings_csv.parent == logger.run_dir
        assert logger.timestamp in logger.runs_csv.name
        assert logger.timestamp in logger.settings_csv.name
    
    def test_logger_creates_run_subdirectory(self, temp_data_dir):
        """Test that logger creates run-specific subdirectory."""
        logger = MaxDiffLogger(data_dir=temp_data_dir)
        
        # Run directory should exist
        assert logger.run_dir.exists()
        assert logger.run_dir.is_dir()
        
        # Should be inside data directory
        assert logger.run_dir.parent == logger.data_dir
        
        # Should contain timestamp in name
        assert logger.timestamp in logger.run_dir.name
    
    def test_logger_with_default_data_dir(self, tmp_path, monkeypatch):
        """Test logger with default data directory."""
        # Change to a temporary directory
        monkeypatch.chdir(tmp_path)
        
        logger = MaxDiffLogger()
        
        assert logger.data_dir.name == "data"
        assert logger.data_dir.exists()
        assert logger.run_dir.exists()
    
    def test_log_environment_settings(self, temp_data_dir):
        """Test logging environment settings."""
        logger = MaxDiffLogger(data_dir=temp_data_dir)
        
        env_vars = {
            'ITEMS_PER_SUBSET': '4',
            'TARGET_TRIALS': '20',
            'OPENAI_API_KEY': 'secret-key',
            'NORMAL_VAR': 'normal-value'
        }
        
        logger.log_environment_settings(env_vars)
        
        assert logger.settings_csv.exists()
        
        # Read and verify contents
        with open(logger.settings_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 1
        row = rows[0]
        
        assert 'timestamp' in row
        assert 'run_id' in row
        assert row['run_id'] == logger.timestamp
        assert row['ITEMS_PER_SUBSET'] == '4'
        assert row['TARGET_TRIALS'] == '20'
        assert row['NORMAL_VAR'] == 'normal-value'
        
        # Sensitive keys should be redacted
        assert row['OPENAI_API_KEY'] == '[REDACTED]'
    
    def test_log_run_results(self, temp_data_dir, sample_task_session, report_config):
        """Test logging run results."""
        logger = MaxDiffLogger(data_dir=temp_data_dir)
        results = aggregate_results(sample_task_session)
        
        logger.log_run_results(sample_task_session, results, report_config, execution_time=120.5)
        
        # Check that main runs CSV exists
        assert logger.runs_csv.exists()
        
        # Check that all CSV files exist
        item_results_csv = logger.run_dir / f"maxdiff_item_results_{logger.timestamp}.csv"
        responses_csv = logger.run_dir / f"maxdiff_responses_{logger.timestamp}.csv"
        
        assert item_results_csv.exists()
        assert responses_csv.exists()
        
        # Verify main runs CSV content
        with open(logger.runs_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 1
        row = rows[0]
        
        assert 'timestamp' in row
        assert 'run_id' in row
        assert row['run_id'] == logger.timestamp
        assert row['total_trials'] == str(results.total_trials)
        assert row['execution_time_seconds'] == '120.5'
        assert 'success_rate' in row
        assert 'models_used' in row
    
    def test_log_item_results(self, temp_data_dir, sample_task_session):
        """Test logging item results."""
        logger = MaxDiffLogger(data_dir=temp_data_dir)
        results = aggregate_results(sample_task_session)
        report_config = ReportConfig()
        
        logger.log_run_results(sample_task_session, results, report_config)
        
        item_results_csv = logger.run_dir / f"maxdiff_item_results_{logger.timestamp}.csv"
        assert item_results_csv.exists()
        
        # Verify content
        with open(item_results_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) > 0
        
        # Check structure of each row
        for row in rows:
            assert 'timestamp' in row
            assert 'run_id' in row
            assert 'item_id' in row
            assert 'item_name' in row
            assert 'consensus_rank' in row
            assert 'utility_score' in row
            assert 'best_count' in row
            assert 'worst_count' in row
    
    def test_log_response_details(self, temp_data_dir, sample_task_session):
        """Test logging response details."""
        logger = MaxDiffLogger(data_dir=temp_data_dir)
        results = aggregate_results(sample_task_session)
        report_config = ReportConfig()
        
        logger.log_run_results(sample_task_session, results, report_config)
        
        responses_csv = logger.run_dir / f"maxdiff_responses_{logger.timestamp}.csv"
        assert responses_csv.exists()
        
        # Verify content
        with open(responses_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == len(sample_task_session.responses)
        
        # Check structure of each row
        for row in rows:
            assert 'timestamp' in row
            assert 'run_id' in row
            assert 'trial_number' in row
            assert 'model_name' in row
            assert 'success' in row
            assert 'best_item_id' in row
            assert 'worst_item_id' in row
            assert 'trial_items' in row
    
    def test_get_run_summary(self, temp_data_dir):
        """Test getting run summary."""
        logger = MaxDiffLogger(data_dir=temp_data_dir)
        
        summary = logger.get_run_summary()
        
        assert isinstance(summary, dict)
        assert 'run_id' in summary
        assert 'runs_csv' in summary
        assert 'settings_csv' in summary
        assert 'item_results_csv' in summary
        assert 'responses_csv' in summary
        
        assert summary['run_id'] == logger.timestamp
        assert str(logger.runs_csv) == summary['runs_csv']
        assert str(logger.settings_csv) == summary['settings_csv']
        
        # Check that all paths point to the run directory
        for key in ['runs_csv', 'settings_csv', 'item_results_csv', 'responses_csv']:
            path = Path(summary[key])
            assert path.parent == logger.run_dir
    
    def test_sensitive_keys_redaction(self, temp_data_dir):
        """Test that sensitive keys are properly redacted."""
        logger = MaxDiffLogger(data_dir=temp_data_dir)
        
        env_vars = {
            'OPENAI_API_KEY': 'sk-123456',
            'ANTHROPIC_API_KEY': 'anthropic-key',
            'GOOGLE_API_KEY': 'google-key',
            'SECRET_PASSWORD': 'password123',
            'API_TOKEN': 'token456',
            'PRIVATE_KEY': 'private789',
            'NORMAL_CONFIG': 'normal-value'
        }
        
        logger.log_environment_settings(env_vars)
        
        # Read and verify redaction
        with open(logger.settings_csv, 'r') as f:
            reader = csv.DictReader(f)
            row = next(reader)
        
        # All sensitive keys should be redacted
        sensitive_keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 
                         'SECRET_PASSWORD', 'API_TOKEN', 'PRIVATE_KEY']
        for key in sensitive_keys:
            assert row[key] == '[REDACTED]'
        
        # Normal config should not be redacted
        assert row['NORMAL_CONFIG'] == 'normal-value'
    
    def test_multiple_loggers_different_timestamps(self, temp_data_dir):
        """Test that multiple loggers get different timestamps/directories."""
        import time
        
        logger1 = MaxDiffLogger(data_dir=temp_data_dir)
        time.sleep(1.1)  # Ensure different timestamp
        logger2 = MaxDiffLogger(data_dir=temp_data_dir)
        
        assert logger1.timestamp != logger2.timestamp
        assert logger1.run_dir != logger2.run_dir
        assert logger1.runs_csv != logger2.runs_csv
    
    def test_csv_file_creation_and_headers(self, temp_data_dir):
        """Test CSV file creation and header writing."""
        logger = MaxDiffLogger(data_dir=temp_data_dir)
        
        # Log some data to trigger file creation
        env_vars = {'TEST_VAR': 'test_value'}
        logger.log_environment_settings(env_vars)
        
        # Check that file was created with proper headers
        with open(logger.settings_csv, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            
        assert 'timestamp' in headers
        assert 'run_id' in headers
        assert 'TEST_VAR' in headers


class TestGetEnvironmentSettings:
    """Test get_environment_settings function."""
    
    def test_get_environment_settings_defaults(self):
        """Test getting environment settings with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            settings = get_environment_settings()
        
        assert isinstance(settings, dict)
        
        # Check that default values are returned
        assert settings['items_per_subset'] == '4'
        assert settings['target_trials'] == '20'
        assert settings['openai_model'] == 'gpt-3.5-turbo'
        assert settings['anthropic_model'] == 'claude-3-haiku-20240307'
        assert settings['google_model'] == 'gemini-1.5-flash'
        assert settings['max_concurrent_requests'] == '5'
        assert settings['output_format'] == 'html'
        
        # Check boolean conversions
        assert settings['has_openai_key'] is False
        assert settings['has_anthropic_key'] is False
        assert settings['has_google_key'] is False
    
    def test_get_environment_settings_with_env_vars(self):
        """Test getting environment settings with environment variables set."""
        env_patch = {
            'ITEMS_PER_SUBSET': '6',
            'TARGET_TRIALS': '30',
            'OPENAI_MODEL': 'gpt-4',
            'OPENAI_API_KEY': 'test-key',
            'MAX_CONCURRENT_REQUESTS': '10',
            'OUTPUT_FORMAT': 'json'
        }
        
        with patch.dict(os.environ, env_patch):
            settings = get_environment_settings()
        
        assert settings['items_per_subset'] == '6'
        assert settings['target_trials'] == '30'
        assert settings['openai_model'] == 'gpt-4'
        assert settings['max_concurrent_requests'] == '10'
        assert settings['output_format'] == 'json'
        assert settings['has_openai_key'] is True
    
    def test_get_environment_settings_system_info(self):
        """Test that system information is included."""
        settings = get_environment_settings()
        
        assert 'python_version' in settings
        assert 'platform' in settings
        assert 'working_directory' in settings
        
        # Verify that actual values are returned
        assert isinstance(settings['python_version'], str)
        assert len(settings['python_version']) > 0
        assert isinstance(settings['working_directory'], str)


class TestIntegration:
    """Test integration scenarios."""
    
    def test_complete_logging_workflow(self, temp_data_dir, sample_task_session):
        """Test complete logging workflow."""
        logger = MaxDiffLogger(data_dir=temp_data_dir)
        
        # 1. Log environment settings
        env_settings = get_environment_settings()
        logger.log_environment_settings(env_settings)
        
        # 2. Log run results
        results = aggregate_results(sample_task_session)
        report_config = ReportConfig(output_format='html', include_raw_responses=True)
        logger.log_run_results(sample_task_session, results, report_config, execution_time=95.2)
        
        # 3. Verify all files exist
        assert logger.settings_csv.exists()
        assert logger.runs_csv.exists()
        
        item_results_csv = logger.run_dir / f"maxdiff_item_results_{logger.timestamp}.csv"
        responses_csv = logger.run_dir / f"maxdiff_responses_{logger.timestamp}.csv"
        assert item_results_csv.exists()
        assert responses_csv.exists()
        
        # 4. Verify run summary
        summary = logger.get_run_summary()
        for file_path in summary.values():
            if file_path != summary['run_id']:  # skip run_id
                assert Path(file_path).exists()
    
    def test_multiple_runs_isolation(self, temp_data_dir, sample_task_session):
        """Test that multiple runs are properly isolated."""
        import time
        
        # Run 1
        logger1 = MaxDiffLogger(data_dir=temp_data_dir)
        results1 = aggregate_results(sample_task_session)
        logger1.log_run_results(sample_task_session, results1, ReportConfig())
        
        time.sleep(1.1)  # Ensure different timestamp
        
        # Run 2
        logger2 = MaxDiffLogger(data_dir=temp_data_dir)
        results2 = aggregate_results(sample_task_session)
        logger2.log_run_results(sample_task_session, results2, ReportConfig())
        
        # Verify separate directories
        assert logger1.run_dir != logger2.run_dir
        assert logger1.run_dir.exists()
        assert logger2.run_dir.exists()
        
        # Verify separate files
        assert logger1.runs_csv != logger2.runs_csv
        assert logger1.runs_csv.exists()
        assert logger2.runs_csv.exists()
        
        # Verify both contain the correct run IDs
        with open(logger1.runs_csv, 'r') as f:
            reader = csv.DictReader(f)
            row1 = next(reader)
            assert row1['run_id'] == logger1.timestamp
        
        with open(logger2.runs_csv, 'r') as f:
            reader = csv.DictReader(f)
            row2 = next(reader)
            assert row2['run_id'] == logger2.timestamp

