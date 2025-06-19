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
    
    # Generate model agreement matrix
    models = list(results.agreement_matrix.keys())
    
    # Helper function to get provider name from model string (reuse from above)
    def get_provider_name_short(model_name):
        if 'openai' in model_name.lower() or 'gpt' in model_name.lower():
            return 'OpenAI'
        elif 'anthropic' in model_name.lower() or 'claude' in model_name.lower():
            return 'Anthropic'
        elif 'google' in model_name.lower() or 'gemini' in model_name.lower():
            return 'Google'
        else:
            return model_name.split('-')[0].capitalize()
    
    agreement_header = "<th>Model</th>" + "".join(f"<th>{get_provider_name_short(model)}</th>" for model in models)
    agreement_rows = ""
    
    for model1 in models:
        model1_provider = get_provider_name_short(model1)
        row = f"<td class='model-name'>{model1_provider}</td>"
        for model2 in models:
            agreement = results.agreement_matrix.get(model1, {}).get(model2, 0)
            if model1 == model2:
                cell_class = "self-agreement"
            elif agreement > 0.7:
                cell_class = "high-agreement"
            elif agreement > 0.4:
                cell_class = "medium-agreement"
            else:
                cell_class = "low-agreement"
            row += f"<td class='{cell_class}'>{agreement:.3f}</td>"
        agreement_rows += f"<tr>{row}</tr>"
    
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
    
    # Generate disagreement points section
    disagreement_cards = ""
    for i, disagreement in enumerate(results.disagreement_points[:5]):  # Show top 5
        trial_items = ", ".join([item['name'] for item in disagreement['items']])
        responses_html = ""
        for resp in disagreement['responses']:
            model_provider = get_provider_name(resp['model'])
            best_item = next((item['name'] for item in disagreement['items'] if item['id'] == resp['best_item_id']), 'Unknown')
            worst_item = next((item['name'] for item in disagreement['items'] if item['id'] == resp['worst_item_id']), 'Unknown')
            # Don't truncate reasoning - show full text
            reasoning_text = resp['reasoning'] if resp['reasoning'] else 'No reasoning provided'
            responses_html += f"""
            <div class="response-item">
                <strong>{model_provider}:</strong> Best: {best_item}, Worst: {worst_item}
                <div class="reasoning">{reasoning_text}</div>
            </div>
            """
        
        disagreement_cards += f"""
        <div class="disagreement-card">
            <h4>Trial {disagreement['trial_number']}</h4>
            <p><strong>Items:</strong> {trial_items}</p>
            <p><strong>Consensus:</strong> Best {disagreement['best_consensus']:.1%}, Worst {disagreement['worst_consensus']:.1%}</p>
            <div class="responses">{responses_html}</div>
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
            <h2>🤝 Model Agreement Matrix</h2>
            <p>How often models agreed on best and worst choices (1.0 = perfect agreement):</p>
            <table>
                <thead>
                    <tr>
                        {agreement_header}
                    </tr>
                </thead>
                <tbody>
                    {agreement_rows}
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>⚡ Major Disagreements</h2>
            <p>Trials where models had significant disagreements (showing top 5):</p>
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

