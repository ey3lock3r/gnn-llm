#!/bin/bash
# APTP-GNN Test Runner
echo "🚀 Starting APTP-GNN v7.1 Test Suite..."

# Run Unit Tests
echo "--- [1/2] Running Unit Tests ---"
uv run python3 tests/unit_tests.py

# Run Integration Tests
echo "--- [2/2] Running Integration Tests ---"
uv run python3 tests/integration_tests.py

echo "✅ All tests completed."
