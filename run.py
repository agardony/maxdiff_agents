#!/usr/bin/env python3
"""
Simple entry point for running the MaxDiff AI agents program.
"""
import sys
import os

# Add the project root to Python path for absolute imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now we can safely import the main module
if __name__ == '__main__':
    from src.main import main
    main()

