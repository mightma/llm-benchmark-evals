#!/usr/bin/env python3
"""
Test script to verify CLI commands are working correctly
"""

import subprocess
import sys

def test_command_help():
    """Test that commands show help without errors."""
    commands_to_test = [
        ["python3", "main.py", "--help"],
        ["python3", "main.py", "evaluate", "--help"],
        ["python3", "main.py", "compare", "--help"],
        ["python3", "main.py", "server-status", "--help"],
        ["python3", "main.py", "list-benchmarks", "--help"],
        ["python3", "main.py", "parameters", "--help"]
    ]

    for cmd in commands_to_test:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✓ {' '.join(cmd[2:])} - Help works")
            else:
                print(f"✗ {' '.join(cmd[2:])} - Error: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"✗ {' '.join(cmd[2:])} - Timeout")
        except Exception as e:
            print(f"✗ {' '.join(cmd[2:])} - Exception: {e}")

if __name__ == "__main__":
    print("Testing CLI commands...")
    test_command_help()
    print("Command test completed!")