#!/bin/bash
# Helper script to run the benchmark and auto-stop all containers when done

set -e

# Parse arguments (pass any additional args to docker-compose)
COMPOSE_ARGS="$@"

echo "Starting benchmark with auto-stop..."
echo "When benchmark completes, all containers will stop automatically."
echo ""

# Run docker-compose with abort-on-container-exit
# This ensures that when benchmark finishes, docker-compose stops all services
docker-compose up --abort-on-container-exit ${COMPOSE_ARGS}

echo ""
echo "Benchmark completed! Cleaning up..."
docker-compose down

echo "Done! Check results/benchmark_results.json for output."

