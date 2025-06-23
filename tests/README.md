# MaxDiff AI Agents Test Suite

This directory contains a comprehensive test suite for the MaxDiff AI Agents project.

## Test Structure

- **`conftest.py`** - Pytest configuration and shared fixtures
- **`test_types.py`** - Tests for data models and type definitions
- **`test_maxdiff_engine.py`** - Tests for the core MaxDiff engine logic
- **`test_model_clients.py`** - Tests for AI model client implementations
- **`test_reporting.py`** - Tests for report generation and data aggregation
- **`test_logging_utils.py`** - Tests for CSV logging and data persistence
- **`test_personas.py`** - Tests for multi-persona functionality and file parsing

## Running Tests

### Using the test runner:
```bash
python run_tests.py
```

### Using pytest directly:
```bash
uv run python -m pytest tests/ -v
```

### Running specific test files:
```bash
uv run python -m pytest tests/test_types.py -v
```

### Running with coverage:
```bash
uv run python -m pytest tests/ --cov=src --cov-report=html
```

## Test Coverage

The test suite provides comprehensive coverage of:

### Core Components (138 tests total)
- **Data Types** - All Pydantic models and validation (18 tests)
- **MaxDiff Engine** - Trial generation, choice recording, completion tracking (15 tests)
- **Model Clients** - API interactions, retry logic, Pydantic response parsing (74 tests)
- **Reporting** - Score calculation, ranking, HTML/JSON generation (25 tests)
- **Logging** - CSV output, run organization, sensitive data redaction (16 tests)
- **Multi-Persona** - File parsing, edge cases, integration testing (20 tests)

### Key Features Tested
- ✅ Input validation and edge cases
- ✅ Error handling and fallback mechanisms  
- ✅ Async operations and retry logic
- ✅ File I/O and data persistence
- ✅ Mock testing for external APIs
- ✅ Data aggregation and statistical calculations
- ✅ Run isolation and timestamp management
- ✅ Multi-persona file parsing and validation
- ✅ Persona priority system and fallback behavior
- ✅ Unicode content and edge case handling

## Test Dependencies

The test suite requires:
- `pytest` - Test framework
- `pytest-asyncio` - For async test support
- `pytest-mock` - Enhanced mocking capabilities

## Test Organization

Tests are organized by module with comprehensive fixtures for:
- Sample data creation
- Temporary file handling
- Mock configurations
- Isolated test environments

Each test class focuses on a specific component with both positive and negative test cases.

## Continuous Integration

All tests must pass before code can be committed. The test suite verifies:
- Functionality correctness
- Error handling robustness
- Performance characteristics
- Data integrity
- Security considerations (API key redaction)

