"""
Simple Test Launcher for Perudo Game AI Project

This module runs all tests when executed. Designed to be launched directly
from PyCharm or command line without any arguments.

Usage:
    python tests/test_launcher.py
"""

import sys
import os
import subprocess
from pathlib import Path


def run_all_tests():
    """Discover and run all tests in the project."""
    # Get paths
    project_root = Path(__file__).parent.parent
    tests_root = Path(__file__).parent

    # Discover all test files
    test_files = []

    # Add foundation test
    foundation_test = tests_root / 'test_foundation_setup.py'
    if foundation_test.exists():
        test_files.append(str(foundation_test))

    # Add unit tests
    unit_dir = tests_root / 'unit'
    if unit_dir.exists():
        test_files.extend([str(f) for f in unit_dir.glob('test_*.py')])

    # Add integration tests
    integration_dir = tests_root / 'integration'
    if integration_dir.exists():
        test_files.extend([str(f) for f in integration_dir.glob('test_*.py')])

    # Add performance tests
    performance_dir = tests_root / 'performance'
    if performance_dir.exists():
        test_files.extend([str(f) for f in performance_dir.glob('test_*.py')])

    # Add regression tests
    regression_dir = tests_root / 'regression'
    if regression_dir.exists():
        test_files.extend([str(f) for f in regression_dir.glob('test_*.py')])

    if not test_files:
        print("No test files found.")
        return 0

    # Print summary
    print(f"Found {len(test_files)} test files")
    print("Running all tests...")
    print("-" * 60)

    # Build and run pytest command
    cmd = [sys.executable, '-m', 'pytest'] + test_files + ['-v', '--color=yes']

    # Change to project root and run
    original_cwd = os.getcwd()
    try:
        os.chdir(project_root)
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode
    finally:
        os.chdir(original_cwd)


if __name__ == '__main__':
    sys.exit(run_all_tests())
