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
    from src.types import TaskSession, ReportConfig, AggregatedResults, ModelResponse
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


def calculate_item_agreement(session: TaskSession) -> List[Dict[str, Any]]:
    """
    Calculate item-level agreement based on utility score standard deviation across models.
    Items with low standard deviation have high agreement, high std dev means low agreement.
    
    Returns:
        List of items sorted by agreement level (low std dev = high agreement first)
    """
    # Calculate item scores for each model
    model_scores = calculate_item_scores_by_model(session)
    item_names = {item.id: item.name for item in session.items}
    
    # Calculate standard deviation of utility scores for each item across models
    item_agreements = []
    
    for item in session.items:
        item_id = item.id
        utility_scores = []
        model_details = []
        
        # Get utility scores for this item from each model
        for model_name, items in model_scores.items():
            if item_id in items and items[item_id]['appearance_count'] > 0:
                utility_score = items[item_id]['utility_score']
                utility_scores.append(utility_score)
                model_details.append({
                    'model': model_name,
                    'utility_score': utility_score,
                    'best_rate': items[item_id]['best_rate'],
                    'worst_rate': items[item_id]['worst_rate'],
                    'appearances': items[item_id]['appearance_count']
                })
        
        # Calculate standard deviation if we have scores from multiple models
        if len(utility_scores) >= 2:
            std_dev = np.std(utility_scores, ddof=1)  # Sample standard deviation
            mean_utility = np.mean(utility_scores)
            agreement_score = 1.0 / (1.0 + std_dev)  # Convert std dev to agreement score (0-1)
            
            item_agreements.append({
                'item_id': item_id,
                'item_name': item_names.get(item_id, 'Unknown'),
                'utility_std_dev': std_dev,
                'mean_utility': mean_utility,
                'agreement_score': agreement_score,
                'model_scores': model_details
            })
    
    # Sort by standard deviation (ascending - low std dev = high agreement first)
    item_agreements.sort(key=lambda x: x['utility_std_dev'])
    
    return item_agreements


def calculate_item_scores_by_model(session: TaskSession) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Calculate item scores for each model separately.
    
    Returns:
        Dict mapping model_name -> item_id -> scoring metrics
    """
    model_scores = defaultdict(lambda: defaultdict(lambda: {
        'best_count': 0,
        'worst_count': 0,
        'appearance_count': 0,
        'best_rate': 0.0,
        'worst_rate': 0.0,
        'utility_score': 0.0
    }))
    
    # Group responses by model
    for response in session.responses:
        if not response.success:
            continue
            
        model_name = response.model_name
        
        # Find the trial this response corresponds to
        trial = next((t for t in session.trials if t.trial_number == response.trial_number), None)
        if not trial:
            continue
            
        # Count appearances for this model
        for item in trial.items:
            model_scores[model_name][item.id]['appearance_count'] += 1
            
        # Count best/worst selections for this model
        model_scores[model_name][response.best_item_id]['best_count'] += 1
        model_scores[model_name][response.worst_item_id]['worst_count'] += 1
    
    # Calculate derived metrics for each model
    for model_name, items in model_scores.items():
        for item_id, scores in items.items():
            appearances = max(scores['appearance_count'], 1)  # Avoid division by zero
            scores['best_rate'] = scores['best_count'] / appearances
            scores['worst_rate'] = scores['worst_count'] / appearances
            scores['utility_score'] = scores['best_rate'] - scores['worst_rate']
    
    return {k: dict(v) for k, v in model_scores.items()}


def identify_disagreement_points(session: TaskSession) -> List[Dict[str, Any]]:
    """
    Identify items with significant disagreement across models based on utility score standard deviation.
    Only items with std dev > 0.2 are considered to have meaningful disagreement.
    
    Returns:
        List of items with significant disagreement, sorted by standard deviation
    """
    # Calculate item scores for each model
    model_scores = calculate_item_scores_by_model(session)
    item_names = {item.id: item.name for item in session.items}
    
    # Define threshold for significant disagreement
    DISAGREEMENT_THRESHOLD = 0.5  # Standard deviation threshold
    
    # Calculate standard deviation of utility scores for each item across models
    item_disagreements = []
    
    for item in session.items:
        item_id = item.id
        utility_scores = []
        model_details = []
        
        # Get utility scores for this item from each model
        for model_name, items in model_scores.items():
            if item_id in items and items[item_id]['appearance_count'] > 0:
                utility_score = items[item_id]['utility_score']
                utility_scores.append(utility_score)
                model_details.append({
                    'model': model_name,
                    'utility_score': utility_score,
                    'best_rate': items[item_id]['best_rate'],
                    'worst_rate': items[item_id]['worst_rate'],
                    'appearances': items[item_id]['appearance_count']
                })
        
        # Calculate standard deviation if we have scores from multiple models
        if len(utility_scores) >= 2:
            std_dev = np.std(utility_scores, ddof=1)  # Sample standard deviation
            mean_utility = np.mean(utility_scores)
            
            # Only include items with significant disagreement
            if std_dev > DISAGREEMENT_THRESHOLD:
                # Find specific examples where this item appeared in trials
                trial_examples = []
                for response in session.responses:
                    if not response.success:
                        continue
                        
                    trial = next((t for t in session.trials if t.trial_number == response.trial_number), None)
                    if trial and any(t.id == item_id for t in trial.items):
                        # Check if this item was chosen as best or worst
                        if response.best_item_id == item_id or response.worst_item_id == item_id:
                            choice_type = 'best' if response.best_item_id == item_id else 'worst'
                            trial_examples.append({
                                'trial_number': response.trial_number,
                                'model': response.model_name,
                                'choice_type': choice_type,
                                'reasoning': response.reasoning,
                                'trial_items': [t.name for t in trial.items]
                            })
                
                item_disagreements.append({
                    'item_id': item_id,
                    'item_name': item_names.get(item_id, 'Unknown'),
                    'utility_std_dev': std_dev,
                    'mean_utility': mean_utility,
                    'model_scores': model_details,
                    'trial_examples': trial_examples[:3]  # Show up to 3 examples
                })
    
    # Sort by standard deviation (highest first) and return items with significant disagreement
    item_disagreements.sort(key=lambda x: x['utility_std_dev'], reverse=True)
    
    return item_disagreements


def aggregate_results(session: TaskSession) -> AggregatedResults:
    """
    Aggregate all results into a comprehensive analysis.
    
    Returns:
        AggregatedResults object
    """
    item_scores = calculate_item_scores(session)
    consensus_ranking = calculate_consensus_ranking(item_scores)
    item_agreement = calculate_item_agreement(session)
    disagreement_points = identify_disagreement_points(session)
    
    models_used = list(set(resp.model_name for resp in session.responses if resp.success))
    
    return AggregatedResults(
        total_trials=len(session.trials),
        models_used=models_used,
        item_scores=item_scores,
        consensus_ranking=consensus_ranking,
        agreement_matrix=item_agreement,  # Now contains item agreement data instead of model matrix
        disagreement_points=disagreement_points
    )


def generate_html_report(session: TaskSession, config: ReportConfig) -> str:
    """Generate HTML format report with modern styling."""
    results = aggregate_results(session)
    item_names = {item.id: item.name for item in session.items}
    
    # Generate consensus ranking table
    ranking_rows = ""
    for i, item_id in enumerate(results.consensus_ranking):
        scores = results.item_scores.get(item_id, {})
        item_name = item_names.get(item_id, 'Unknown')
        utility_score = scores.get('utility_score', 0)
        best_rate = scores.get('best_rate', 0)
        worst_rate = scores.get('worst_rate', 0)
        
        # Color coding based on utility score
        if utility_score > 0.3:
            row_class = "high-score"
        elif utility_score > 0:
            row_class = "medium-score"
        else:
            row_class = "low-score"
            
        ranking_rows += f"""
        <tr class="{row_class}">
            <td class="rank">{i + 1}</td>
            <td class="item-name">{item_name}</td>
            <td class="score">{utility_score:.3f}</td>
            <td class="rate">{best_rate:.1%}</td>
            <td class="rate">{worst_rate:.1%}</td>
        </tr>
        """
    
    # Helper function to get provider name from model string
    def get_provider_name(model_name):
        if 'openai' in model_name.lower() or 'gpt' in model_name.lower():
            return 'OpenAI'
        elif 'anthropic' in model_name.lower() or 'claude' in model_name.lower():
            return 'Anthropic'
        elif 'google' in model_name.lower() or 'gemini' in model_name.lower():
            return 'Google'
        else:
            return model_name.split('-')[0].capitalize()
    
    # Create a mapping from item_id to consensus rank
    consensus_rank_map = {item_id: i + 1 for i, item_id in enumerate(results.consensus_ranking)}
    
    # Generate item agreement list (sorted by std dev - low std dev = high agreement)
    agreement_rows = ""
    for i, item_agreement in enumerate(results.agreement_matrix):  # Show all items
        item_name = item_agreement['item_name']
        item_id = item_agreement['item_id']
        std_dev = item_agreement['utility_std_dev']
        mean_utility = item_agreement['mean_utility']
        agreement_score = item_agreement['agreement_score']
        
        # Get consensus rank for this item
        consensus_rank = consensus_rank_map.get(item_id, 'N/A')
        
        # Color coding based on agreement level (low std dev = high agreement)
        if std_dev < 0.1:
            row_class = "high-agreement"
        elif std_dev < 0.3:
            row_class = "medium-agreement"
        else:
            row_class = "low-agreement"
            
        # Model scores details
        model_details = ", ".join([
            f"{get_provider_name(score['model'])}: {score['utility_score']:.2f}"
            for score in item_agreement['model_scores']
        ])
            
        agreement_rows += f"""
        <tr class="{row_class}">
            <td class="rank">{consensus_rank}</td>
            <td class="rank">{i + 1}</td>
            <td class="item-name">{item_name}</td>
            <td class="score">{std_dev:.3f}</td>
            <td class="score">{mean_utility:.3f}</td>
            <td class="model-details">{model_details}</td>
        </tr>
        """
    
    # Generate disagreement points section (items with highest utility score variance)
    disagreement_cards = ""
    for i, disagreement in enumerate(results.disagreement_points[:3]):  # Show top 3
        item_name = disagreement['item_name']
        std_dev = disagreement['utility_std_dev']
        mean_utility = disagreement['mean_utility']
        
        # Model scores section
        model_scores_html = ""
        for model_score in disagreement['model_scores']:
            model_provider = get_provider_name(model_score['model'])
            utility = model_score['utility_score']
            best_rate = model_score['best_rate']
            worst_rate = model_score['worst_rate']
            appearances = model_score['appearances']
            
            # Color code based on utility score
            if utility > 0.3:
                score_class = "high-score"
            elif utility > 0:
                score_class = "medium-score"
            else:
                score_class = "low-score"
                
            model_scores_html += f"""
            <div class="model-score-item {score_class}">
                <strong>{model_provider}:</strong> 
                Utility: {utility:.3f} (Best: {best_rate:.1%}, Worst: {worst_rate:.1%})
                <span class="appearances">({appearances} appearances)</span>
            </div>
            """
        
        # Trial examples section
        examples_html = ""
        if disagreement['trial_examples']:
            examples_html = "<h5>Example Choices:</h5>"
            for example in disagreement['trial_examples']:
                model_provider = get_provider_name(example['model'])
                choice_type = example['choice_type']
                reasoning = example['reasoning'] if example['reasoning'] else 'No reasoning provided'
                trial_items = ", ".join(example['trial_items'])
                
                choice_class = "best-choice" if choice_type == "best" else "worst-choice"
                examples_html += f"""
                <div class="trial-example {choice_class}">
                    <strong>{model_provider}</strong> chose as <em>{choice_type}</em> in trial with: {trial_items}
                    <div class="reasoning">{reasoning}</div>
                </div>
                """
        
        disagreement_cards += f"""
        <div class="disagreement-card">
            <h4>{item_name}</h4>
            <p><strong>Utility Score Variance:</strong> σ = {std_dev:.3f} (Mean: {mean_utility:.3f})</p>
            <p>This item shows the {i+1}{'st' if i+1==1 else 'nd' if i+1==2 else 'rd'} highest disagreement across AI models.</p>
            
            <h5>Model Scores:</h5>
            <div class="model-scores">{model_scores_html}</div>
            
            {examples_html}
        </div>
        """
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MaxDiff AI Agents Results</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}
        
        .summary-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: transform 0.3s ease;
        }}
        
        .summary-card:hover {{
            transform: translateY(-5px);
        }}
        
        .summary-card .number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #4facfe;
            margin-bottom: 10px;
        }}
        
        .summary-card .label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .section {{
            padding: 40px;
            border-bottom: 1px solid #eee;
        }}
        
        .section:last-child {{
            border-bottom: none;
        }}
        
        .section h2 {{
            font-size: 1.8em;
            margin-bottom: 25px;
            color: #333;
            border-left: 4px solid #4facfe;
            padding-left: 15px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        
        .rank {{
            font-weight: bold;
            font-size: 1.1em;
            text-align: center;
            width: 60px;
        }}
        
        .item-name {{
            font-weight: 500;
        }}
        
        .score, .rate {{
            text-align: center;
            font-family: 'Courier New', monospace;
        }}
        
        .high-score {{
            background: linear-gradient(90deg, #d4edda, transparent);
        }}
        
        .medium-score {{
            background: linear-gradient(90deg, #fff3cd, transparent);
        }}
        
        .low-score {{
            background: linear-gradient(90deg, #f8d7da, transparent);
        }}
        
        .self-agreement {{
            background: #e9ecef;
            font-weight: bold;
        }}
        
        .high-agreement {{
            background: #d4edda;
            color: #155724;
        }}
        
        .medium-agreement {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .low-agreement {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .model-name {{
            font-weight: 500;
        }}
        
        .disagreement-cards {{
            display: grid;
            gap: 20px;
            margin-top: 20px;
        }}
        
        .disagreement-card {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.05);
        }}
        
        .disagreement-card h4 {{
            color: #4facfe;
            margin-bottom: 10px;
        }}
        
        .responses {{
            margin-top: 15px;
        }}
        
        .response-item {{
            background: #f8f9fa;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 3px solid #4facfe;
        }}
        
        .reasoning {{
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
            font-style: italic;
        }}
        
        .models-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }}
        
        .model-tag {{
            background: #4facfe;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.9em;
        }}
        
        .model-scores {{
            margin: 15px 0;
        }}
        
        .model-score-item {{
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 3px solid #4facfe;
        }}
        
        .model-score-item.high-score {{
            background: #d4edda;
            border-left-color: #28a745;
        }}
        
        .model-score-item.medium-score {{
            background: #fff3cd;
            border-left-color: #ffc107;
        }}
        
        .model-score-item.low-score {{
            background: #f8d7da;
            border-left-color: #dc3545;
        }}
        
        .appearances {{
            font-size: 0.8em;
            color: #666;
            margin-left: 10px;
        }}
        
        .trial-example {{
            background: #f8f9fa;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 3px solid #6c757d;
        }}
        
        .trial-example.best-choice {{
            border-left-color: #28a745;
            background: #d4f6d4;
        }}
        
        .trial-example.worst-choice {{
            border-left-color: #dc3545;
            background: #f8d7da;
        }}
        
        .disagreement-card h5 {{
            margin: 15px 0 10px 0;
            color: #495057;
            font-size: 1.1em;
        }}
        
        .model-details {{
            font-size: 0.9em;
            font-family: 'Courier New', monospace;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                margin: 10px;
                border-radius: 10px;
            }}
            
            .header {{
                padding: 20px;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .section {{
                padding: 20px;
            }}
            
            table {{
                font-size: 0.9em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🍎 MaxDiff AI Agents Results</h1>
            <div class="subtitle">Consensus Rankings & Model Analysis</div>
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <div class="number">{results.total_trials}</div>
                <div class="label">Total Trials</div>
            </div>
            <div class="summary-card">
                <div class="number">{len(results.models_used)}</div>
                <div class="label">AI Models</div>
            </div>
            <div class="summary-card">
                <div class="number">{len(session.items)}</div>
                <div class="label">Items Evaluated</div>
            </div>
            <div class="summary-card">
                <div class="number">{len(results.disagreement_points)}</div>
                <div class="label">Disagreements</div>
            </div>
        </div>
        
        <div class="section">
            <h2>🏆 Consensus Ranking</h2>
            <p>Items ranked by utility score (best rate - worst rate):</p>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Item</th>
                        <th>Utility Score</th>
                        <th>Best Rate</th>
                        <th>Worst Rate</th>
                    </tr>
                </thead>
                <tbody>
                    {ranking_rows}
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>🤖 AI Models Used</h2>
            <div class="models-list">
                {' '.join(f'<span class="model-tag">{get_provider_name(model)}</span>' for model in results.models_used)}
            </div>
        </div>
        
        <div class="section">
            <h2>🤝 Item Agreement Analysis</h2>
            <p>Items ranked by agreement level (low standard deviation = high agreement):</p>
            <table>
                <thead>
                    <tr>
                        <th>Consensus Rank</th>
                        <th>Agreement Rank</th>
                        <th>Item</th>
                        <th>Std Dev (σ)</th>
                        <th>Mean Utility</th>
                        <th>Model Scores</th>
                    </tr>
                </thead>
                <tbody>
                    {agreement_rows}
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>⚡ Major Disagreements</h2>
            <p>Items with highest utility score variance across AI models (top 3 most controversial):</p>
            <div class="disagreement-cards">
                {disagreement_cards}
            </div>
        </div>
    </div>
</body>
</html>
    """
    
    return html_content


def generate_json_report(session: TaskSession, config: ReportConfig) -> str:
    """Generate JSON format report (kept for API/programmatic use)."""
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
    
    # Item agreement analysis
    console.print("\\n[bold green]Item Agreement Analysis[/bold green]")
    agreement_table = Table()
    agreement_table.add_column("Rank", width=6)
    agreement_table.add_column("Item", min_width=15)
    agreement_table.add_column("Std Dev", width=10)
    agreement_table.add_column("Mean Utility", width=12)
    
    for i, item_agreement in enumerate(results.agreement_matrix[:5]):  # Show top 5 most agreed items
        agreement_table.add_row(
            str(i + 1),
            item_agreement['item_name'],
            f"{item_agreement['utility_std_dev']:.3f}",
            f"{item_agreement['mean_utility']:.3f}"
        )
    
    console.print(agreement_table)
    
    # Disagreement summary
    console.print(f"\\n[bold red]Major Disagreements:[/bold red] {len(results.disagreement_points)} items with high utility score variance")


def generate_report(session: TaskSession, config: ReportConfig):
    """
    Generate and save the final report.
    """
    # Always print console summary
    print_console_summary(session)
    
    if config.output_format == 'html':
        report_content = generate_html_report(session, config)
        if config.output_file:
            with open(config.output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"\\n📄 HTML Report saved to: {config.output_file}")
        else:
            print("\\n" + report_content)
    elif config.output_format == 'json':
        report_content = generate_json_report(session, config)
        if config.output_file:
            with open(config.output_file, 'w') as f:
                f.write(report_content)
            print(f"\\n📄 JSON Report saved to: {config.output_file}")
        else:
            print("\\n" + report_content)
    else:
        print(f"\\nUnsupported output format: {config.output_format}. Using HTML instead.")
        report_content = generate_html_report(session, config)
        output_file = config.output_file.replace('.json', '.html') if config.output_file else 'report.html'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"\\n📄 HTML Report saved to: {output_file}")

