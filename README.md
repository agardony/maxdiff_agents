# MaxDiff AI Agents

A Python project that evaluates MaxDiff tasks using multiple frontier AI models (OpenAI, Anthropic, Google) to understand consensus and disagreement patterns in preference evaluation.

## Overview

This project implements a MaxDiff (Maximum Difference Scaling) survey methodology using AI models to evaluate items. It runs trials in parallel across multiple AI providers and provides detailed analysis of consensus rankings and disagreement patterns between models.

## Features

- **Multi-Model Support**: OpenAI GPT, Anthropic Claude, and Google Gemini
- **Async Execution**: Parallel processing of trials across all models
- **Configurable Parameters**: All MaxDiff settings controllable via `.env` file
- **Rich Reporting**: Detailed analysis with consensus rankings and disagreement detection
- **Structured Output**: JSON reports with optional raw response inclusion

## Installation

1. **Clone or create the project directory**
2. **Set up virtual environment with uv**:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys and preferences
   ```

## Configuration

Edit your `.env` file with the following settings:

### API Keys (at least one required)
```bash
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### MaxDiff Configuration
```bash
ITEMS_PER_SUBSET=4          # Number of items shown per trial
TARGET_TRIALS=20            # Total number of trials to run
DIMENSION_POSITIVE_LABEL=Best
DIMENSION_NEGATIVE_LABEL=Worst
INSTRUCTION_TEXT=Please choose the item you find BEST and the item you find WORST.
PERSONA=You are an expert evaluating these items objectively
```

#### Persona Examples
Customize the AI model's perspective with different personas:
- `PERSONA=You are an expert marketing product manager`
- `PERSONA=You are a software engineer evaluating development tools`
- `PERSONA=You are a UX designer focusing on user experience`
- `PERSONA=You are a business executive prioritizing strategic initiatives`
- `PERSONA=You are a customer evaluating products for purchase`

### Model Configuration
```bash
OPENAI_MODEL=gpt-3.5-turbo
ANTHROPIC_MODEL=claude-3-haiku-20240307
GOOGLE_MODEL=gemini-1.5-flash
```

**Note**: Default models are selected for cost-effectiveness. For higher quality responses, consider:
- `OPENAI_MODEL=gpt-4` or `gpt-4-turbo`
- `ANTHROPIC_MODEL=claude-3-sonnet-20240229` or `claude-3-opus-20240229`
- `GOOGLE_MODEL=gemini-pro` or `gemini-1.5-pro`

### Performance & Output
```bash
MAX_CONCURRENT_REQUESTS=5
REQUEST_TIMEOUT=30
OUTPUT_FORMAT=json
INCLUDE_RAW_RESPONSES=false
REPORT_STYLE=detailed
```

## Usage

### Basic Usage
```bash
# Activate environment
source .venv/bin/activate

# Run with default settings
python max_diff.py

# Run with custom items file
python max_diff.py --items-file my_items.txt

# Run with custom environment file
python max_diff.py --env-file .env.production
```

### Input Format
Create a text file with one item per line:
```
Reliability
Innovation
Cost-effectiveness
User experience
Security
Performance
```

## Output

The program generates multiple types of output:

### 1. Console Summary
Rich terminal output with tables showing rankings and agreement analysis.

### 2. HTML/JSON Reports
Detailed structured output including:
- Consensus ranking with utility scores
- Item-level metrics (best/worst rates, net scores)
- Model agreement matrix
- Disagreement analysis
- Optional raw responses

### 3. Comprehensive CSV Logging
Automatic timestamped CSV files saved to the `data/` directory:

- **`maxdiff_runs_YYYYMMDD_HHMMSS.csv`**: Main run results with:
  - Execution metrics (trials, models, success rates)
  - Top/bottom items and utility scores
  - Agreement statistics
  - Performance data

- **`maxdiff_settings_YYYYMMDD_HHMMSS.csv`**: Environment configuration (API keys redacted for security)

- **`maxdiff_item_results_YYYYMMDD_HHMMSS.csv`**: Detailed item-level data:
  - Individual item rankings and scores
  - Model-specific utility scores
  - Agreement metrics per item

- **`maxdiff_responses_YYYYMMDD_HHMMSS.csv`**: Response-level details:
  - Individual trial responses
  - Best/worst choices per model
  - Error tracking
  - Reasoning analysis

**Privacy Note**: API keys and sensitive environment variables are automatically redacted from CSV logs.

### Example Output Structure
```json
{
  "summary": {
    "total_trials": 20,
    "models_used": ["openai-gpt-4", "anthropic-claude-3-opus-20240229"],
    "items_evaluated": 15
  },
  "consensus_ranking": [
    {
      "rank": 1,
      "item_name": "Security",
      "utility_score": 0.425
    }
  ],
  "model_agreement": {
    "openai-gpt-4": {
      "anthropic-claude-3-opus-20240229": 0.65
    }
  }
}
```

## Architecture

### Core Components

- **`types.py`**: Pydantic models and type definitions
- **`maxdiff_engine.py`**: Trial generation and choice recording logic
- **`model_clients.py`**: AI model client implementations
- **`reporting.py`**: Analysis and report generation
- **`main.py`**: Orchestration and CLI interface

### MaxDiff Methodology

The implementation follows standard MaxDiff principles:
- Random subset generation for each trial
- Best/worst choice recording
- Utility score calculation: `best_rate - worst_rate`
- Consensus ranking based on utility scores

## API Keys Setup

### OpenAI
1. Visit https://platform.openai.com/api-keys
2. Create a new API key
3. Add to `.env` as `OPENAI_API_KEY`

### Anthropic
1. Visit https://console.anthropic.com/
2. Generate an API key
3. Add to `.env` as `ANTHROPIC_API_KEY`

### Google
1. Visit https://makersuite.google.com/app/apikey
2. Create an API key
3. Add to `.env` as `GOOGLE_API_KEY`

## Development

### Project Structure
```
maxdiff_agents/
├── src/
│   ├── __init__.py
│   ├── types.py
│   ├── maxdiff_engine.py
│   ├── model_clients.py
│   ├── reporting.py
│   ├── logging_utils.py
│   └── main.py
├── data/                   # CSV logs (git-ignored)
│   ├── maxdiff_runs_*.csv
│   ├── maxdiff_settings_*.csv
│   ├── maxdiff_item_results_*.csv
│   └── maxdiff_responses_*.csv
├── .env.example
├── requirements.txt
├── setup.py
├── setup.cfg
├── max_diff.py
└── sample_items.txt
```

### Extending the Project

- **Add new AI providers**: Implement `ModelClient` interface
- **Custom analysis**: Extend reporting functions
- **Output formats**: Add HTML/Markdown generators
- **Advanced sampling**: Modify trial generation in engine

## Troubleshooting

**No API keys found**: Ensure at least one API key is configured in `.env`

**Model errors**: Check API key validity and model names

**Import errors**: Ensure virtual environment is activated and dependencies installed

**Rate limiting**: Adjust `MAX_CONCURRENT_REQUESTS` and `REQUEST_TIMEOUT`

## License

MIT License - feel free to modify and extend for your needs.

