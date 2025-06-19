"""
Main entry for MaxDiff AI agents program.
"""
import os
import asyncio
import click
import time
from dotenv import load_dotenv
# Import handling for both package and standalone execution
try:
    # Try relative imports first (when run as package)
    from .types import MaxDiffItem, EngineConfig, TaskSession, ReportConfig
    from .maxdiff_engine import MaxDiffEngine
    from .model_clients import OpenAIClient, AnthropicClient, GoogleClient
    from .reporting import generate_report, aggregate_results
    from .logging_utils import MaxDiffLogger, get_environment_settings
except ImportError:
    # Fallback to absolute imports (when run standalone via max_diff.py)
    from src.types import MaxDiffItem, EngineConfig, TaskSession, ReportConfig
    from src.maxdiff_engine import MaxDiffEngine
    from src.model_clients import OpenAIClient, AnthropicClient, GoogleClient
    from src.reporting import generate_report, aggregate_results
    from src.logging_utils import MaxDiffLogger, get_environment_settings


@click.command()
@click.option('--items-file', default='sample_items.txt', help='File containing items for evaluation.')
@click.option('--env-file', default='.env', help='Environment configuration file.')
def main(items_file: str, env_file: str):
    """
    Main function to execute MaxDiff tasks with AI models.
    """
    # Load environment configuration
    load_dotenv(env_file)
    
    # Initialize logger
    logger = MaxDiffLogger()
    
    # Log environment settings (excluding sensitive keys)
    env_settings = get_environment_settings()
    logger.log_environment_settings(env_settings)
    
    print(f"📊 Logging to data directory with run ID: {logger.timestamp}")
    
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
    
    print(f"Initialized {len(clients)} model clients:")
    for client in clients:
        print(f"  - {client.model_name}")
    
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
                tasks.append((task, trial, client))
        
        print(f"Created {len(tasks)} tasks ({len(trials)} trials × {len(clients)} clients)")
        
        # Wait for all tasks to complete
        for i, (task, trial, client) in enumerate(tasks):
            try:
                response = await task
                responses.append(response)
                print(f"Completed task {i+1}/{len(tasks)} - Trial {trial.trial_number}, Client {client.model_name}")
            except Exception as e:
                print(f"Error in task {i+1}/{len(tasks)} - Trial {trial.trial_number}, Client {client.model_name}: {e}")
        
        return responses
    
    # Run the async task loop with timing
    start_time = time.time()
    responses = asyncio.run(execute_trials())
    execution_time = time.time() - start_time
    
    print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
    
    # Compile task session
    session = TaskSession(
        items=items,
        config=config,
        trials=trials,
        responses=responses
    )
    
    # Generate report
    report_config = ReportConfig(
        output_format=os.getenv('OUTPUT_FORMAT', 'html'),
        include_raw_responses=os.getenv('INCLUDE_RAW_RESPONSES', 'false').lower() == 'true',
        report_style=os.getenv('REPORT_STYLE', 'detailed'),
        output_file=os.getenv('REPORT_OUTPUT_FILE', 'report.html')
    )
    
    generate_report(session, report_config)
    
    # Log comprehensive results to CSV
    results = aggregate_results(session)
    logger.log_run_results(session, results, report_config, execution_time)
    
    # Print logging summary
    log_summary = logger.get_run_summary()
    print("\n📁 CSV Files Created:")
    print(f"   • Main Results: {log_summary['runs_csv']}")
    print(f"   • Settings: {log_summary['settings_csv']}")
    print(f"   • Item Details: {log_summary['item_results_csv']}")
    print(f"   • Response Details: {log_summary['responses_csv']}")


if __name__ == '__main__':
    main()

