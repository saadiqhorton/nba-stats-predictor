#!/usr/bin/env bash
# Gracefully reload nginx configuration without dropping connections
# Usage: ./scripts/reload_nginx.sh

set -euo pipefail

NGINX_CONTAINER="nba-nginx"

# Verify Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: docker command not found. Please install Docker."
    exit 1
fi

if ! docker info &> /dev/null 2>&1; then
    echo "Error: Docker daemon is not running. Please start Docker."
    exit 1
fi

# Verify the container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${NGINX_CONTAINER}$"; then
    echo "Error: Nginx container '${NGINX_CONTAINER}' is not running."
    echo "Start the stack first: docker compose up -d"
    exit 1
fi

# Test nginx config before reloading
echo "Testing nginx configuration..."
if ! docker exec "$NGINX_CONTAINER" nginx -t 2>&1; then
    echo "Error: Nginx configuration test failed. Fix the config before reloading."
    exit 1
fi

# Reload nginx gracefully (existing connections stay open)
echo "Reloading nginx..."
docker exec "$NGINX_CONTAINER" nginx -s reload

echo "Nginx reloaded successfully."
echo "Active connections were not interrupted."
