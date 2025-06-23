"""
Main entry for MaxDiff AI agents program.
"""
import os
import asyncio
import click
import time
from typing import List, Optional
from dotenv import load_dotenv
# Import handling for both package and standalone execution
try:
    # Try relative imports first (when run as package)
    from .types import MaxDiffItem, EngineConfig, TaskSession, ReportConfig, ModelConfig
    from .maxdiff_engine import MaxDiffEngine
    from .model_clients import OpenAIClient, AnthropicClient, GoogleClient
    from .reporting import generate_report, aggregate_results
    from .logging_utils import MaxDiffLogger, get_environment_settings
except ImportError:
    # Fallback to absolute imports (when run standalone via max_diff.py)
    from src.types import MaxDiffItem, EngineConfig, TaskSession, ReportConfig, ModelConfig
    from src.maxdiff_engine import MaxDiffEngine
    from src.model_clients import OpenAIClient, AnthropicClient, GoogleClient
    from src.reporting import generate_report, aggregate_results
from src.logging_utils import MaxDiffLogger, get_environment_settings


def load_personas(personas_file: Optional[str], single_persona: Optional[str]) -> List[str]:
    """
    Load personas from file or use single persona or fallback to .env PERSONA.
    
    Args:
        personas_file: Path to file containing personas delimited by === PERSONA N ===
        single_persona: Single persona string to use (overrides all others)
        
    Returns:
        List of persona strings
    """
    # Priority 1: Single persona parameter
    if single_persona:
        return [single_persona.strip()]
    
    # Priority 2: Personas file
    if personas_file:
        try:
            with open(personas_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                print(f"Warning: Personas file '{personas_file}' is empty.")
                return []
            
            # Split by delimiter pattern
            import re
            delimiter_pattern = r'^\s*===\s*PERSONA\s+\d+\s*===\s*$'
            
            # Find all delimiter positions
            lines = content.split('\n')
            delimiter_indices = []
            
            for i, line in enumerate(lines):
                if re.match(delimiter_pattern, line, re.IGNORECASE):
                    delimiter_indices.append(i)
            
            if not delimiter_indices:
                # No delimiters found - treat entire file as single persona
                print(f"Warning: No valid delimiters found in '{personas_file}'. Treating entire file as single persona.")
                return [content.strip()] if content.strip() else []
            
            # Extract personas between delimiters
            personas = []
            for i, start_idx in enumerate(delimiter_indices):
                # Determine end index
                if i + 1 < len(delimiter_indices):
                    end_idx = delimiter_indices[i + 1]
                else:
                    end_idx = len(lines)
                
                # Extract persona content (skip the delimiter line)
                persona_lines = lines[start_idx + 1:end_idx]
                persona_content = '\n'.join(persona_lines).strip()
                
                if persona_content:
                    personas.append(persona_content)
                else:
                    print(f"Warning: Empty persona found after delimiter at line {start_idx + 1}")
            
            if not personas:
                print(f"Warning: No valid personas found in '{personas_file}'.")
                return []
            
            print(f"Loaded {len(personas)} persona(s) from '{personas_file}'")
            return personas
            
        except FileNotFoundError:
            print(f"Error: Personas file '{personas_file}' not found.")
            return []
        except Exception as e:
            print(f"Error reading personas file '{personas_file}': {e}")
            return []
    
    # Priority 3: Fallback to .env PERSONA
    env_persona = os.getenv('PERSONA')
    if env_persona:
        print("Using persona from .env PERSONA variable")
        return [env_persona.strip()]
    
    # No personas found
    print("Warning: No personas specified. Use --personas-file, --persona, or set PERSONA in .env")
    return []


@click.command()
@click.option('--items-file', default='sample_items.txt', help='File containing items for evaluation.')
@click.option('--env-file', default='.env', help='Environment configuration file.')
@click.option('--personas-file', default=None, help='File containing personas delimited by === PERSONA N ===.')
@click.option('--persona', default=None, help='Single persona to use (overrides personas-file and .env PERSONA).')
def main(items_file: str, env_file: str, personas_file: Optional[str], persona: Optional[str]):
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
    
    # Load personas
    personas = load_personas(personas_file, persona)
    if not personas:
        print("Error: No personas found. Please provide a personas file with at least one persona.")
        return
    
    # Load base configuration from .env (persona will be set per iteration)
    base_config = EngineConfig(
        items_per_subset=int(os.getenv('ITEMS_PER_SUBSET', 4)),
        target_trials=int(os.getenv('TARGET_TRIALS', 20)),
        dimension_positive_label=os.getenv('DIMENSION_POSITIVE_LABEL', 'Best'),
        dimension_negative_label=os.getenv('DIMENSION_NEGATIVE_LABEL', 'Worst'),
        instruction_text=os.getenv('INSTRUCTION_TEXT', 'Please choose the item you find BEST and the item you find WORST.'),
        persona=personas[0]  # Will be updated per persona iteration
    )
    
    # Load items
    with open(items_file, 'r') as f:
        items = [
            MaxDiffItem(name=line.strip()) for line in f if line.strip()
        ]
    
    # Initialize the engine
    engine = MaxDiffEngine(items=items, config=base_config)
    trials = engine.generate_all_trials()
    
    # Load model configuration from environment
    model_config = ModelConfig(
        max_retries=int(os.getenv('MAX_RETRIES', 3)),
        retry_base_delay=float(os.getenv('RETRY_BASE_DELAY', 1.0)),
        retry_max_delay=float(os.getenv('RETRY_MAX_DELAY', 60.0)),
        request_timeout=int(os.getenv('REQUEST_TIMEOUT', 30))
    )
    
    print(f"📡 Retry configuration: max_retries={model_config.max_retries}, base_delay={model_config.retry_base_delay}s, max_delay={model_config.retry_max_delay}s")
    
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
    
    # Execute trials asynchronously with multiple personas
    async def execute_trials():
        all_responses = []
        max_concurrent_requests = int(os.getenv('MAX_CONCURRENT_REQUESTS', 5))
        semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        async def execute_single_trial_model_persona(trial, client, persona, persona_index):
            async with semaphore:
                # Create config with current persona
                current_config = EngineConfig(
                    items_per_subset=base_config.items_per_subset,
                    target_trials=base_config.target_trials,
                    dimension_positive_label=base_config.dimension_positive_label,
                    dimension_negative_label=base_config.dimension_negative_label,
                    instruction_text=base_config.instruction_text,
                    persona=persona
                )
                response = await client.evaluate_trial(trial, current_config)
                # Add persona information to response
                response.persona_index = persona_index
                return response
        
        # Create all tasks for all personas
        tasks = []
        for persona_index, persona in enumerate(personas):
            print(f"Setting up tasks for persona {persona_index + 1}/{len(personas)}")
            for trial in trials:
                for client in clients:
                    task = asyncio.create_task(execute_single_trial_model_persona(trial, client, persona, persona_index))
                    tasks.append((task, trial, client, persona_index))
        
        total_tasks = len(tasks)
        print(f"Created {total_tasks} tasks ({len(trials)} trials × {len(clients)} clients × {len(personas)} personas)")
        
        # Wait for all tasks to complete
        for i, (task, trial, client, persona_index) in enumerate(tasks):
            try:
                response = await task
                all_responses.append(response)
                print(f"Completed task {i+1}/{total_tasks} - Persona {persona_index + 1}, Trial {trial.trial_number}, Client {client.model_name}")
            except Exception as e:
                print(f"Error in task {i+1}/{total_tasks} - Persona {persona_index + 1}, Trial {trial.trial_number}, Client {client.model_name}: {e}")
        
        return all_responses
    
    # Run the async task loop with timing
    start_time = time.time()
    responses = asyncio.run(execute_trials())
    execution_time = time.time() - start_time
    
    print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
    
    # Compile task session (using base config for consistency)
    session = TaskSession(
        items=items,
        config=base_config,
        trials=trials,
        responses=responses
    )
    
    # Add personas information to session for reporting
    session.personas = personas
    
    # Generate report
    report_config = ReportConfig(
        output_format=os.getenv('OUTPUT_FORMAT', 'html'),
        include_raw_responses=os.getenv('INCLUDE_RAW_RESPONSES', 'false').lower() == 'true',
        report_style=os.getenv('REPORT_STYLE', 'detailed'),
        output_file=os.getenv('REPORT_OUTPUT_FILE', 'report.html')
    )
    
    # Generate report with timestamp matching CSV logging
    report_file = generate_report(session, report_config, timestamp=logger.timestamp)
    
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

