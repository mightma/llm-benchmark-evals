#!/usr/bin/env python3
"""
Simple test script to verify CLI command parsing works correctly
"""

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock the required modules that might not be installed
import importlib.util

def create_mock_module(name, content=""):
    """Create a mock module."""
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    if content:
        exec(content, module.__dict__)
    sys.modules[name] = module
    return module

# Mock httpx
mock_httpx = create_mock_module('httpx', '''
class AsyncClient:
    def __init__(self, timeout=None):
        pass
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass
    async def get(self, url):
        class MockResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {"data": [{"id": "test-model"}]}
        return MockResponse()

HTTPError = Exception
''')

# Mock other dependencies
create_mock_module('datasets', 'def load_dataset(*args, **kwargs): return []')
create_mock_module('transformers')
create_mock_module('torch')

try:
    # Now test the CLI
    from main import cli, async_command

    # Test that the async_command decorator works
    @async_command
    async def test_async_function():
        await asyncio.sleep(0.1)
        return "success"

    result = test_async_function()
    assert result == "success"

    print("✓ async_command decorator works correctly")

    # Test CLI command registration
    commands = cli.list_commands(None)
    expected_commands = ['evaluate', 'compare', 'server-status', 'list-benchmarks', 'parameters']

    for cmd in expected_commands:
        if cmd in commands:
            print(f"✓ Command '{cmd}' registered successfully")
        else:
            print(f"✗ Command '{cmd}' not found")

    print("\n✓ CLI structure test completed successfully!")

except Exception as e:
    print(f"✗ CLI test failed: {e}")
    import traceback
    traceback.print_exc()