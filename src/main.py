"""
Main entry for MaxDiff AI agents program.
"""
import os
import asyncio
import click
from dotenv import load_dotenv
try:
    from .types import MaxDiffItem, EngineConfig, TaskSession, ReportConfig
    from .maxdiff_engine import MaxDiffEngine
    from .model_clients import OpenAIClient, AnthropicClient, GoogleClient
    from .reporting import generate_report
except ImportError:
    from types import MaxDiffItem, EngineConfig, TaskSession, ReportConfig
    from maxdiff_engine import MaxDiffEngine
    from model_clients import OpenAIClient, AnthropicClient, GoogleClient
    from reporting import generate_report


@click.command()
@click.option('--items-file', default='sample_items.txt', help='File containing items for evaluation.')
@click.option('--env-file', default='.env', help='Environment configuration file.')
def main(items_file: str, env_file: str):
    """
    Main function to execute MaxDiff tasks with AI models.
    """
    # Load environment configuration
    load_dotenv(env_file)
    
    # Load configuration from .env
    config = EngineConfig(
        items_per_subset=int(os.getenv('ITEMS_PER_SUBSET', 4)),
        target_trials=int(os.getenv('TARGET_TRIALS', 20)),
        dimension_positive_label=os.getenv('DIMENSION_POSITIVE_LABEL', 'Best'),
        dimension_negative_label=os.getenv('DIMENSION_NEGATIVE_LABEL', 'Worst'),
        instruction_text=os.getenv('INSTRUCTION_TEXT', 'Please choose the item you find BEST and the item you find WORST.'),
        persona=os.getenv('PERSONA', 'You are an expert evaluating these items objectively')
    )
    
    # Load items
    with open(items_file, 'r') as f:
        items = [
            MaxDiffItem(name=line.strip()) for line in f if line.strip()
        ]
    
    # Initialize the engine
    engine = MaxDiffEngine(items=items, config=config)
    trials = engine.generate_all_trials()
    
    # Initialize AI model clients
    clients = []
    
    # Add OpenAI client if API key is available
    if os.getenv('OPENAI_API_KEY'):
        clients.append(OpenAIClient(
            model_name=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
            api_key=os.getenv('OPENAI_API_KEY')
        ))
    
    # Add Anthropic client if API key is available
    if os.getenv('ANTHROPIC_API_KEY'):
        clients.append(AnthropicClient(
            model_name=os.getenv('ANTHROPIC_MODEL', 'claude-3-haiku-20240307'),
            api_key=os.getenv('ANTHROPIC_API_KEY')
        ))
    
    # Add Google client if API key is available
    if os.getenv('GOOGLE_API_KEY'):
        clients.append(GoogleClient(
            model_name=os.getenv('GOOGLE_MODEL', 'gemini-1.5-flash'),
            api_key=os.getenv('GOOGLE_API_KEY')
        ))
    
    if not clients:
        print("Error: No API keys found. Please configure at least one model API key in your .env file.")
        return
    
    # Execute trials asynchronously
    async def execute_trials():
        responses = []
        max_concurrent_requests = int(os.getenv('MAX_CONCURRENT_REQUESTS', 5))
        semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        async def execute_single_trial_model(trial, client):
            async with semaphore:
                return await client.evaluate_trial(trial, config)
        
        # Create all tasks
        tasks = []
        for trial in trials:
            for client in clients:
                task = asyncio.create_task(execute_single_trial_model(trial, client))
                tasks.append((task, trial))
        
        # Wait for all tasks to complete
        for task, trial in tasks:
            response = await task
            # Find the corresponding items for recording the choice
            best_item = next((item for item in trial.items if item.id == response.best_item_id), None)
            worst_item = next((item for item in trial.items if item.id == response.worst_item_id), None)
            
            if best_item and worst_item:
                try:
                    engine.record_choice(
                        presented_items=trial.items,
                        best_item=best_item,
                        worst_item=worst_item,
                        model_name=response.model_name,
                        reasoning=response.reasoning
                    )
                except Exception as e:
                    print(f"Warning: Could not record choice for trial {trial.trial_number}, model {response.model_name}: {e}")
            
            responses.append(response)
        return responses
    
    # Run the async task loop
    responses = asyncio.run(execute_trials())
    
    # Compile task session
    session = TaskSession(
        items=items,
        config=config,
        trials=trials,
        responses=responses
    )
    
    # Generate report
    report_config = ReportConfig(
        output_format=os.getenv('OUTPUT_FORMAT', 'json'),
        include_raw_responses=os.getenv('INCLUDE_RAW_RESPONSES', 'false').lower() == 'true',
        report_style=os.getenv('REPORT_STYLE', 'detailed'),
        output_file=os.getenv('REPORT_OUTPUT_FILE', 'report.json')
    )
    
    generate_report(session, report_config)


if __name__ == '__main__':
    main()

