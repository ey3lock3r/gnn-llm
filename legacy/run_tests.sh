#!/bin/bash
set -e

echo "🚀 Starting APTP-GigaGraph v8.2.2 Test Suite..."
export PYTHONPATH=.

echo "--- [1/3] Running Unit Tests ---"
uv run python3 tests/unit_tests.py

echo "--- [2/3] Running Integration Tests ---"
uv run python3 tests/integration_tests.py

echo "--- [3/3] Running Checkpoint Persistence Tests ---"
uv run python3 tests/checkpoint_tests.py

echo "✅ ALL GIGA-TESTS PASSED (v8.2.2)"
