#!/usr/bin/env bash
# Run on your laptop: bundle exactly what the Docker image needs (no weights, no outputs).
set -euo pipefail
cd "$(dirname "$0")/.."
tar czf caremate_service.tar.gz Dockerfile .dockerignore requirements-service.txt \
    caremate/__init__.py caremate/perception_v0_7.py caremate/workspace_v0_7.py caremate/enrol_v0_7.py \
    caremate/service/__init__.py caremate/service/app.py references/v0_7 deploy/ec2_bootstrap.sh
echo "Created $(pwd)/caremate_service.tar.gz ($(du -h caremate_service.tar.gz | cut -f1))"
