"""
Reporting and analysis module for MaxDiff results.
"""
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
try:
    from .types import TaskSession, ReportConfig, AggregatedResults, ModelResponse
except ImportError:
    from types import TaskSession, ReportConfig, AggregatedResults, ModelResponse
from rich.console import Console
from rich.table import Table
from rich.text import Text


def calculate_item_scores(session: TaskSession) -> Dict[str, Dict[str, float]]:
    """
    Calculate various scoring metrics for each item based on MaxDiff responses.
    
    Returns:
        Dict mapping item_id to scoring metrics
    """
    item_scores = defaultdict(lambda: {
        'best_count': 0,
        'worst_count': 0,
        'appearance_count': 0,
        'best_rate': 0.0,
        'worst_rate': 0.0,
        'net_score': 0.0,
        'utility_score': 0.0
    })
    
    # Count best/worst selections and appearances
    for response in session.responses:
        if not response.success:
            continue
            
        # Find the trial this response corresponds to
        trial = next((t for t in session.trials if t.trial_number == response.trial_number), None)
        if not trial:
            continue
            
        # Count appearances
        for item in trial.items:
            item_scores[item.id]['appearance_count'] += 1
            
        # Count best/worst selections
        item_scores[response.best_item_id]['best_count'] += 1
        item_scores[response.worst_item_id]['worst_count'] += 1
    
    # Calculate derived metrics
    for item_id, scores in item_scores.items():
        appearances = max(scores['appearance_count'], 1)  # Avoid division by zero
        scores['best_rate'] = scores['best_count'] / appearances
        scores['worst_rate'] = scores['worst_count'] / appearances
        scores['net_score'] = scores['best_count'] - scores['worst_count']
        scores['utility_score'] = scores['best_rate'] - scores['worst_rate']
    
    return dict(item_scores)


def calculate_consensus_ranking(item_scores: Dict[str, Dict[str, float]]) -> List[str]:
    """
    Calculate consensus ranking based on utility scores.
    
    Returns:
        List of item_ids ranked from best to worst
    """
    items_with_scores = [
        (item_id, scores['utility_score']) 
        for item_id, scores in item_scores.items()
    ]
    # Sort by utility score (descending)
    items_with_scores.sort(key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in items_with_scores]


def calculate_model_agreement(session: TaskSession) -> Dict[str, Dict[str, float]]:
    """
    Calculate agreement matrix between different models.
    
    Returns:
        Dict mapping model1 -> model2 -> agreement_score
    """
    # Group responses by trial and model
    responses_by_trial = defaultdict(dict)
    models = set()
    
    for response in session.responses:
        if response.success:
            responses_by_trial[response.trial_number][response.model_name] = response
            models.add(response.model_name)
    
    agreement_matrix = defaultdict(lambda: defaultdict(float))
    
    # Calculate pairwise agreement
    for model1 in models:
        for model2 in models:
            if model1 == model2:
                agreement_matrix[model1][model2] = 1.0
                continue
                
            agreements = 0
            total_comparisons = 0
            
            for trial_num, trial_responses in responses_by_trial.items():
                if model1 in trial_responses and model2 in trial_responses:
                    resp1 = trial_responses[model1]
                    resp2 = trial_responses[model2]
                    
                    # Agreement if both models chose the same best and worst items
                    if (resp1.best_item_id == resp2.best_item_id and 
                        resp1.worst_item_id == resp2.worst_item_id):
                        agreements += 1
                    total_comparisons += 1
            
            if total_comparisons > 0:
                agreement_matrix[model1][model2] = agreements / total_comparisons
    
    return {k: dict(v) for k, v in agreement_matrix.items()}


def identify_disagreement_points(session: TaskSession) -> List[Dict[str, Any]]:
    """
    Identify specific trials where models strongly disagreed.
    
    Returns:
        List of disagreement points with details
    """
    responses_by_trial = defaultdict(list)
    
    for response in session.responses:
        if response.success:
            responses_by_trial[response.trial_number].append(response)
    
    disagreements = []
    
    for trial_num, trial_responses in responses_by_trial.items():
        if len(trial_responses) < 2:
            continue
            
        # Find the trial details
        trial = next((t for t in session.trials if t.trial_number == trial_num), None)
        if not trial:
            continue
        
        # Count unique best/worst choices
        best_choices = Counter(resp.best_item_id for resp in trial_responses)
        worst_choices = Counter(resp.worst_item_id for resp in trial_responses)
        
        # Consider it a disagreement if there's no consensus on best or worst
        best_consensus = max(best_choices.values()) / len(trial_responses)
        worst_consensus = max(worst_choices.values()) / len(trial_responses)
        
        if best_consensus < 0.67 or worst_consensus < 0.67:  # Less than 2/3 agreement
            disagreement = {
                'trial_number': trial_num,
                'items': [{'id': item.id, 'name': item.name} for item in trial.items],
                'best_choices': dict(best_choices),
                'worst_choices': dict(worst_choices),
                'best_consensus': best_consensus,
                'worst_consensus': worst_consensus,
                'responses': [
                    {
                        'model': resp.model_name,
                        'best_item_id': resp.best_item_id,
                        'worst_item_id': resp.worst_item_id,
                        'reasoning': resp.reasoning
                    }
                    for resp in trial_responses
                ]
            }
            disagreements.append(disagreement)
    
    # Sort by level of disagreement (lowest consensus first)
    disagreements.sort(key=lambda x: min(x['best_consensus'], x['worst_consensus']))
    
    return disagreements


def aggregate_results(session: TaskSession) -> AggregatedResults:
    """
    Aggregate all results into a comprehensive analysis.
    
    Returns:
        AggregatedResults object
    """
    item_scores = calculate_item_scores(session)
    consensus_ranking = calculate_consensus_ranking(item_scores)
    agreement_matrix = calculate_model_agreement(session)
    disagreement_points = identify_disagreement_points(session)
    
    models_used = list(set(resp.model_name for resp in session.responses if resp.success))
    
    return AggregatedResults(
        total_trials=len(session.trials),
        models_used=models_used,
        item_scores=item_scores,
        consensus_ranking=consensus_ranking,
        agreement_matrix=agreement_matrix,
        disagreement_points=disagreement_points
    )


def generate_json_report(session: TaskSession, config: ReportConfig) -> str:
    """Generate JSON format report."""
    results = aggregate_results(session)
    
    # Add item names to make the report more readable
    item_names = {item.id: item.name for item in session.items}
    
    report_data = {
        'summary': {
            'total_trials': results.total_trials,
            'models_used': results.models_used,
            'items_evaluated': len(session.items)
        },
        'consensus_ranking': [
            {
                'rank': i + 1,
                'item_id': item_id,
                'item_name': item_names.get(item_id, 'Unknown'),
                'utility_score': results.item_scores.get(item_id, {}).get('utility_score', 0)
            }
            for i, item_id in enumerate(results.consensus_ranking)
        ],
        'item_scores': {
            item_names.get(item_id, item_id): scores 
            for item_id, scores in results.item_scores.items()
        },
        'model_agreement': results.agreement_matrix,
        'disagreement_points': results.disagreement_points
    }
    
    if config.include_raw_responses:
        report_data['raw_responses'] = [
            {
                'model_name': resp.model_name,
                'trial_number': resp.trial_number,
                'best_item_id': resp.best_item_id,
                'worst_item_id': resp.worst_item_id,
                'reasoning': resp.reasoning,
                'raw_response': resp.raw_response,
                'success': resp.success,
                'error_message': resp.error_message
            }
            for resp in session.responses
        ]
    
    return json.dumps(report_data, indent=2)


def print_console_summary(session: TaskSession):
    """Print a rich console summary of the results."""
    console = Console()
    results = aggregate_results(session)
    item_names = {item.id: item.name for item in session.items}
    
    # Title
    console.print("\\n[bold blue]MaxDiff AI Agents Results Summary[/bold blue]\\n")
    
    # Basic stats
    console.print(f"📊 [bold]Total Trials:[/bold] {results.total_trials}")
    console.print(f"🤖 [bold]Models Used:[/bold] {', '.join(results.models_used)}")
    console.print(f"📝 [bold]Items Evaluated:[/bold] {len(session.items)}\\n")
    
    # Consensus ranking table
    ranking_table = Table(title="Consensus Ranking", show_header=True, header_style="bold magenta")
    ranking_table.add_column("Rank", width=6)
    ranking_table.add_column("Item", min_width=20)
    ranking_table.add_column("Utility Score", width=12)
    ranking_table.add_column("Best Rate", width=10)
    ranking_table.add_column("Worst Rate", width=10)
    
    for i, item_id in enumerate(results.consensus_ranking[:10]):  # Show top 10
        scores = results.item_scores.get(item_id, {})
        ranking_table.add_row(
            str(i + 1),
            item_names.get(item_id, item_id),
            f"{scores.get('utility_score', 0):.3f}",
            f"{scores.get('best_rate', 0):.3f}",
            f"{scores.get('worst_rate', 0):.3f}"
        )
    
    console.print(ranking_table)
    
    # Model agreement
    console.print("\\n[bold green]Model Agreement Matrix[/bold green]")
    agreement_table = Table()
    agreement_table.add_column("Model")
    
    models = list(results.agreement_matrix.keys())
    for model in models:
        agreement_table.add_column(model, width=10)
    
    for model1 in models:
        row = [model1]
        for model2 in models:
            agreement = results.agreement_matrix.get(model1, {}).get(model2, 0)
            row.append(f"{agreement:.3f}")
        agreement_table.add_row(*row)
    
    console.print(agreement_table)
    
    # Disagreement summary
    console.print(f"\\n[bold red]Major Disagreements:[/bold red] {len(results.disagreement_points)} trials with significant disagreement")


def generate_report(session: TaskSession, config: ReportConfig):
    """
    Generate and save the final report.
    """
    # Always print console summary
    print_console_summary(session)
    
    if config.output_format == 'json':
        report_content = generate_json_report(session, config)
        if config.output_file:
            with open(config.output_file, 'w') as f:
                f.write(report_content)
            print(f"\\n📄 Report saved to: {config.output_file}")
        else:
            print("\\n" + report_content)
    
    # Add other output formats (HTML, Markdown) here in the future

