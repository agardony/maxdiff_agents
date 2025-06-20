"""
Tests for the reporting module.
"""
import pytest
import json
import tempfile
import os
from unittest.mock import patch

from src.reporting import (
    calculate_item_scores, calculate_consensus_ranking, 
    calculate_item_agreement, calculate_item_scores_by_model,
    identify_disagreement_points, aggregate_results,
    generate_json_report, generate_html_report, generate_report
)
from src.types import TaskSession, ReportConfig, AggregatedResults


class TestCalculateItemScores:
    """Test item score calculation."""
    
    def test_calculate_item_scores_basic(self, sample_task_session):
        """Test basic item score calculation."""
        scores = calculate_item_scores(sample_task_session)
        
        # Should have scores for all items that appeared in trials
        assert len(scores) > 0
        
        # Check structure of scores
        for item_id, item_scores in scores.items():
            assert 'best_count' in item_scores
            assert 'worst_count' in item_scores
            assert 'appearance_count' in item_scores
            assert 'best_rate' in item_scores
            assert 'worst_rate' in item_scores
            assert 'net_score' in item_scores
            assert 'utility_score' in item_scores
            
            # Utility score should be best_rate - worst_rate
            expected_utility = item_scores['best_rate'] - item_scores['worst_rate']
            assert abs(item_scores['utility_score'] - expected_utility) < 0.001
            
            # Net score should be best_count - worst_count
            expected_net = item_scores['best_count'] - item_scores['worst_count']
            assert item_scores['net_score'] == expected_net
    
    def test_calculate_item_scores_with_failed_responses(self, sample_task_session):
        """Test item score calculation with failed responses."""
        # Add a failed response
        from src.types import ModelResponse
        failed_response = ModelResponse(
            model_name="failed-model",
            trial_number=1,
            best_item_id="invalid",
            worst_item_id="invalid",
            success=False,
            error_message="Test failure"
        )
        sample_task_session.responses.append(failed_response)
        
        scores = calculate_item_scores(sample_task_session)
        
        # Failed responses should be ignored
        assert len(scores) > 0
        # All scores should still be valid
        for item_scores in scores.values():
            assert item_scores['best_count'] >= 0
            assert item_scores['worst_count'] >= 0


class TestConsensusRanking:
    """Test consensus ranking calculation."""
    
    def test_calculate_consensus_ranking(self, sample_task_session):
        """Test consensus ranking calculation."""
        scores = calculate_item_scores(sample_task_session)
        ranking = calculate_consensus_ranking(scores)
        
        assert isinstance(ranking, list)
        assert len(ranking) == len(scores)
        
        # Check that ranking is ordered by utility score (descending)
        for i in range(len(ranking) - 1):
            current_utility = scores[ranking[i]]['utility_score']
            next_utility = scores[ranking[i + 1]]['utility_score']
            assert current_utility >= next_utility
    
    def test_consensus_ranking_uniqueness(self, sample_task_session):
        """Test that consensus ranking contains unique items."""
        scores = calculate_item_scores(sample_task_session)
        ranking = calculate_consensus_ranking(scores)
        
        assert len(ranking) == len(set(ranking))  # All unique
        assert set(ranking) == set(scores.keys())  # Contains all items


class TestItemAgreement:
    """Test item agreement calculation."""
    
    def test_calculate_item_agreement(self, sample_task_session):
        """Test item agreement calculation."""
        agreement = calculate_item_agreement(sample_task_session)
        
        assert isinstance(agreement, list)
        
        # Check structure of agreement data
        for item_data in agreement:
            assert 'item_id' in item_data
            assert 'item_name' in item_data
            assert 'utility_std_dev' in item_data
            assert 'mean_utility' in item_data
            assert 'agreement_score' in item_data
            assert 'model_scores' in item_data
            
            # Agreement score should be between 0 and 1
            assert 0 <= item_data['agreement_score'] <= 1
            
            # Standard deviation should be non-negative
            assert item_data['utility_std_dev'] >= 0
    
    def test_item_agreement_sorting(self, sample_task_session):
        """Test that item agreement is sorted by std dev (ascending)."""
        agreement = calculate_item_agreement(sample_task_session)
        
        if len(agreement) > 1:
            for i in range(len(agreement) - 1):
                current_std = agreement[i]['utility_std_dev']
                next_std = agreement[i + 1]['utility_std_dev']
                assert current_std <= next_std


class TestItemScoresByModel:
    """Test item scores by model calculation."""
    
    def test_calculate_item_scores_by_model(self, sample_task_session):
        """Test item scores calculation by model."""
        model_scores = calculate_item_scores_by_model(sample_task_session)
        
        assert isinstance(model_scores, dict)
        
        # Should have scores for each model that provided successful responses
        successful_models = set(r.model_name for r in sample_task_session.responses if r.success)
        assert set(model_scores.keys()) == successful_models
        
        # Check structure for each model
        for model_name, items in model_scores.items():
            assert isinstance(items, dict)
            for item_id, scores in items.items():
                assert 'best_count' in scores
                assert 'worst_count' in scores
                assert 'appearance_count' in scores
                assert 'best_rate' in scores
                assert 'worst_rate' in scores
                assert 'utility_score' in scores


class TestDisagreementPoints:
    """Test disagreement point identification."""
    
    def test_identify_disagreement_points(self, sample_task_session):
        """Test disagreement point identification."""
        disagreements = identify_disagreement_points(sample_task_session)
        
        assert isinstance(disagreements, list)
        
        # Check structure of disagreement data
        for disagreement in disagreements:
            assert 'item_id' in disagreement
            assert 'item_name' in disagreement
            assert 'utility_std_dev' in disagreement
            assert 'mean_utility' in disagreement
            assert 'model_scores' in disagreement
            assert 'trial_examples' in disagreement
            
            # Should be sorted by std dev (descending)
            assert disagreement['utility_std_dev'] >= 0
    
    def test_disagreement_points_sorting(self, sample_task_session):
        """Test that disagreement points are sorted by std dev (descending)."""
        disagreements = identify_disagreement_points(sample_task_session)
        
        if len(disagreements) > 1:
            for i in range(len(disagreements) - 1):
                current_std = disagreements[i]['utility_std_dev']
                next_std = disagreements[i + 1]['utility_std_dev']
                assert current_std >= next_std


class TestAggregateResults:
    """Test result aggregation."""
    
    def test_aggregate_results(self, sample_task_session):
        """Test result aggregation."""
        results = aggregate_results(sample_task_session)
        
        assert isinstance(results, AggregatedResults)
        assert results.total_trials == len(sample_task_session.trials)
        assert len(results.models_used) > 0
        assert len(results.item_scores) > 0
        assert len(results.consensus_ranking) > 0
        assert isinstance(results.agreement_matrix, list)
        assert isinstance(results.disagreement_points, list)
        
        # Models used should match successful responses
        successful_models = set(r.model_name for r in sample_task_session.responses if r.success)
        assert set(results.models_used) == successful_models
        
        # Consensus ranking should contain all scored items
        assert set(results.consensus_ranking) == set(results.item_scores.keys())


class TestJSONReport:
    """Test JSON report generation."""
    
    def test_generate_json_report_basic(self, sample_task_session, report_config):
        """Test basic JSON report generation."""
        report_json = generate_json_report(sample_task_session, report_config)
        
        # Should be valid JSON
        report_data = json.loads(report_json)
        
        # Check structure
        assert 'summary' in report_data
        assert 'consensus_ranking' in report_data
        assert 'item_scores' in report_data
        assert 'model_agreement' in report_data
        assert 'disagreement_points' in report_data
        
        # Summary structure
        summary = report_data['summary']
        assert 'total_trials' in summary
        assert 'models_used' in summary
        assert 'items_evaluated' in summary
        
        # Consensus ranking structure
        for rank_item in report_data['consensus_ranking']:
            assert 'rank' in rank_item
            assert 'item_id' in rank_item
            assert 'item_name' in rank_item
            assert 'utility_score' in rank_item
    
    def test_generate_json_report_with_raw_responses(self, sample_task_session):
        """Test JSON report with raw responses included."""
        config = ReportConfig(include_raw_responses=True)
        report_json = generate_json_report(sample_task_session, config)
        
        report_data = json.loads(report_json)
        assert 'raw_responses' in report_data
        
        # Check raw responses structure
        for response in report_data['raw_responses']:
            assert 'model_name' in response
            assert 'trial_number' in response
            assert 'success' in response


class TestHTMLReport:
    """Test HTML report generation."""
    
    def test_generate_html_report_basic(self, sample_task_session, report_config):
        """Test basic HTML report generation."""
        html_content = generate_html_report(sample_task_session, report_config)
        
        assert isinstance(html_content, str)
        assert '<!DOCTYPE html>' in html_content
        assert '<html' in html_content
        assert '</html>' in html_content
        
        # Check for key sections
        assert 'MaxDiff AI Agents Results' in html_content
        assert 'Combined Rankings' in html_content
        assert 'AI Models Used' in html_content
        assert 'Major Disagreements' in html_content
        
        # Should contain CSS styling
        assert '<style>' in html_content
        assert '</style>' in html_content
    
    def test_html_report_contains_data(self, sample_task_session, report_config):
        """Test that HTML report contains actual data."""
        html_content = generate_html_report(sample_task_session, report_config)
        
        # Should contain item names from the sample data
        item_names = [item.name for item in sample_task_session.items]
        for name in item_names[:3]:  # Check first few items
            assert name in html_content
        
        # Should contain model names
        model_names = set(r.model_name for r in sample_task_session.responses if r.success)
        for model_name in model_names:
            # HTML might modify model names for display
            model_provider = model_name.split('-')[0] if '-' in model_name else model_name
            assert model_provider.lower() in html_content.lower()


class TestReportGeneration:
    """Test main report generation function."""
    
    def test_generate_report_json_to_file(self, sample_task_session, tmp_path):
        """Test generating JSON report to file."""
        output_file = tmp_path / "test_report.json"
        config = ReportConfig(
            output_format='json',
            output_file=str(output_file)
        )
        
        with patch('src.reporting.print_console_summary'):
            result_file = generate_report(sample_task_session, config, timestamp="20240101_120000")
        
        # File should be created with timestamp
        expected_file = tmp_path / "test_report_20240101_120000.json"
        assert expected_file.exists()
        assert result_file == str(expected_file)
        
        # Verify file contents
        with open(expected_file, 'r') as f:
            report_data = json.load(f)
        
        assert 'summary' in report_data
        assert 'consensus_ranking' in report_data
    
    def test_generate_report_html_to_file(self, sample_task_session, tmp_path):
        """Test generating HTML report to file."""
        output_file = tmp_path / "test_report.html"
        config = ReportConfig(
            output_format='html',
            output_file=str(output_file)
        )
        
        with patch('src.reporting.print_console_summary'):
            result_file = generate_report(sample_task_session, config, timestamp="20240101_120000")
        
        # File should be created with timestamp
        expected_file = tmp_path / "test_report_20240101_120000.html"
        assert expected_file.exists()
        assert result_file == str(expected_file)
        
        # Verify file contents
        with open(expected_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        assert '<!DOCTYPE html>' in html_content
        assert 'MaxDiff AI Agents Results' in html_content
    
    def test_generate_report_unsupported_format(self, sample_task_session, tmp_path):
        """Test generating report with unsupported format."""
        output_file = tmp_path / "test_report.pdf"
        # Create a config that bypasses pydantic validation
        config = ReportConfig(
            output_format='json',  # Valid format initially
            output_file=str(output_file)
        )
        # Modify after creation to bypass validation
        config.output_format = 'pdf'  # Now set to unsupported format
        
        with patch('src.reporting.print_console_summary'):
            result_file = generate_report(sample_task_session, config, timestamp="20240101_120000")
        
        # According to the implementation, it should replace .json with .html in the filename
        # But since we have .pdf, the implementation will create a timestamped html file
        expected_file = tmp_path / "test_report_20240101_120000.html"
        assert expected_file.exists()
        assert result_file == str(expected_file)
    
    def test_generate_report_with_timestamp_naming(self, sample_task_session, tmp_path):
        """Test that report files are generated with timestamp naming."""
        output_file = tmp_path / "test_report.html"
        config = ReportConfig(
            output_format='html',
            output_file=str(output_file)
        )
        
        timestamp = "20240101_120000"
        
        with patch('src.reporting.print_console_summary'):
            result_file = generate_report(sample_task_session, config, timestamp=timestamp)
        
        # Check that the result file has the timestamp in the name
        expected_file = tmp_path / "test_report_20240101_120000.html"
        assert expected_file.exists()
        assert result_file == str(expected_file)
        
        # Verify file contents
        with open(expected_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        assert '<!DOCTYPE html>' in html_content
        assert 'MaxDiff AI Agents Results' in html_content
    
    def test_generate_report_json_with_timestamp_naming(self, sample_task_session, tmp_path):
        """Test that JSON report files are generated with timestamp naming."""
        output_file = tmp_path / "test_report.json"
        config = ReportConfig(
            output_format='json',
            output_file=str(output_file)
        )
        
        timestamp = "20240101_120000"
        
        with patch('src.reporting.print_console_summary'):
            result_file = generate_report(sample_task_session, config, timestamp=timestamp)
        
        # Check that the result file has the timestamp in the name
        expected_file = tmp_path / "test_report_20240101_120000.json"
        assert expected_file.exists()
        assert result_file == str(expected_file)
        
        # Verify file contents
        with open(expected_file, 'r') as f:
            report_data = json.load(f)
        
        assert 'summary' in report_data
        assert 'consensus_ranking' in report_data
    
    def test_generate_report_default_filename_with_timestamp(self, sample_task_session):
        """Test default filename generation with timestamp."""
        config = ReportConfig(
            output_format='html',
            output_file=None  # No file specified
        )
        
        timestamp = "20240101_120000"
        
        with patch('src.reporting.print_console_summary'):
            result_file = generate_report(sample_task_session, config, timestamp=timestamp)
        
        # When no output file is specified, should return None (prints to console)
        assert result_file is None


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_responses(self, sample_items, engine_config):
        """Test handling of task session with no responses."""
        from src.types import TaskSession, TrialSet
        
        # Create session with trials but no responses
        trials = [TrialSet(trial_number=1, items=sample_items[:4])]
        session = TaskSession(
            items=sample_items,
            config=engine_config,
            trials=trials,
            responses=[]
        )
        
        # Should not crash
        scores = calculate_item_scores(session)
        assert isinstance(scores, dict)
        
        results = aggregate_results(session)
        assert isinstance(results, AggregatedResults)
        assert results.total_trials == 1
        assert len(results.models_used) == 0
    
    def test_single_model_responses(self, sample_task_session):
        """Test handling with only one model's responses."""
        # Filter to only one model
        single_model_responses = [
            r for r in sample_task_session.responses 
            if r.model_name == sample_task_session.responses[0].model_name
        ]
        sample_task_session.responses = single_model_responses
        
        # Should still work
        agreement = calculate_item_agreement(sample_task_session)
        disagreements = identify_disagreement_points(sample_task_session)
        
        # With only one model, agreement calculation might be limited
        assert isinstance(agreement, list)
        assert isinstance(disagreements, list)
    
    def test_invalid_trial_references(self, sample_task_session):
        """Test handling of responses with invalid trial references."""
        from src.types import ModelResponse
        
        # Add response with invalid trial number
        invalid_response = ModelResponse(
            model_name="test-model",
            trial_number=999,  # Non-existent trial
            best_item_id=sample_task_session.items[0].id,
            worst_item_id=sample_task_session.items[1].id,
            success=True
        )
        sample_task_session.responses.append(invalid_response)
        
        # Should handle gracefully
        scores = calculate_item_scores(sample_task_session)
        assert isinstance(scores, dict)
        
        results = aggregate_results(sample_task_session)
        assert isinstance(results, AggregatedResults)

