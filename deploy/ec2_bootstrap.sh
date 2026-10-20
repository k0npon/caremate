#!/usr/bin/env bash
# Run ON the EC2 instance (Ubuntu 24.04) after copying caremate_service.tar.gz to it:
#   tar xzf caremate_service.tar.gz && bash deploy/ec2_bootstrap.sh
# Installs Docker (first time), builds the image, (re)starts the service on port 8000.
set -euo pipefail
if ! command -v docker >/dev/null; then
  sudo apt-get update -y
  sudo apt-get install -y docker.io
  sudo systemctl enable --now docker
fi
TOKEN_FILE="$HOME/.caremate_token"; ADMIN_FILE="$HOME/.caremate_admin_token"
for f in "$TOKEN_FILE" "$ADMIN_FILE"; do
  [ -f "$f" ] || { openssl rand -hex 16 > "$f"; chmod 600 "$f"; }
done
sudo docker build -t caremate-vision:0.7 .
sudo docker rm -f caremate >/dev/null 2>&1 || true
# PUBLIC=1 leaves /v1/perceive open (for judges); otherwise a bearer token is required.
TOKEN_ENV=(-e CAREMATE_API_TOKEN="$(cat "$TOKEN_FILE")")
[ "${PUBLIC:-0}" = "1" ] && TOKEN_ENV=()
# Enrolled references live in a named volume, so they survive rebuilds/restarts.
sudo docker run -d --name caremate --restart unless-stopped -p 80:8000 \
  --memory 3g --cpus 2 --log-opt max-size=20m --log-opt max-file=3 \
  -v caremate_refs:/data/references -e CAREMATE_ADMIN_TOKEN="$(cat "$ADMIN_FILE")" \
  "${TOKEN_ENV[@]}" caremate-vision:0.7
echo "Waiting for the service..."
for i in $(seq 1 60); do curl -fs localhost/v1/health && echo && break; sleep 2; done
[ "${PUBLIC:-0}" = "1" ] || echo "API token (robots, keep private): $(cat "$TOKEN_FILE")"
echo "Admin token (staff enrolment, keep private): $(cat "$ADMIN_FILE")"
echo "Logs: sudo docker logs -f caremate"
