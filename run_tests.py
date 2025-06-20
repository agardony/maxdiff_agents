#!/usr/bin/env python3
"""
Test runner for MaxDiff AI agents.
"""
import sys
import subprocess
import os
from pathlib import Path

def main():
    """Run the test suite."""
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Install test dependencies if needed
    try:
        import pytest
        import pytest_asyncio
    except ImportError:
        print("Installing test dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "-r", "tests/requirements.txt"
        ], check=True)
    
    # Run tests
    test_args = [
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker validation
        "tests/"  # Test directory
    ]
    
    # Add any command line arguments
    test_args.extend(sys.argv[1:])
    
    # Execute pytest
    exit_code = subprocess.run([sys.executable, "-m", "pytest"] + test_args).returncode
    sys.exit(exit_code)

if __name__ == "__main__":
    main()

