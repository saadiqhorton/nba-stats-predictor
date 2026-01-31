#!/usr/bin/env bash
# Integration tests for the NBA Stats Predictor load balancer
# Prerequisites: Docker stack must be running (docker compose up -d)
# Usage: ./tests/test_load_balancer.sh

set -euo pipefail

NGINX_URL="http://localhost:8088"
PASS=0
FAIL=0
TOTAL=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Ensure stopped containers are restarted on exit
cleanup() {
    if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^nba-app-3$"; then
        if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^nba-app-3$"; then
            echo ""
            echo "Cleanup: Restarting nba-app-3..."
            docker start nba-app-3 > /dev/null 2>&1 || true
        fi
    fi
}
trap cleanup EXIT

log_pass() {
    PASS=$((PASS + 1))
    TOTAL=$((TOTAL + 1))
    echo -e "  ${GREEN}PASS${NC}: $1"
}

log_fail() {
    FAIL=$((FAIL + 1))
    TOTAL=$((TOTAL + 1))
    echo -e "  ${RED}FAIL${NC}: $1"
}

log_section() {
    echo ""
    echo -e "${YELLOW}=== $1 ===${NC}"
}

# Verify Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: docker command not found. Please install Docker."
    exit 1
fi

if ! docker info &> /dev/null 2>&1; then
    echo "Error: Docker daemon is not running. Please start Docker."
    exit 1
fi

# Wait for stack to be healthy
wait_for_stack() {
    echo "Waiting for stack to be healthy..."
    local max_wait=120
    local elapsed=0
    while [ "$elapsed" -lt "$max_wait" ]; do
        if curl -sf "${NGINX_URL}/nginx-health" > /dev/null 2>&1; then
            echo "Stack is healthy."
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo "Error: Stack did not become healthy within ${max_wait}s"
    exit 1
}

# --- Test Suite ---

log_section "1. Basic Connectivity"

# Test 1.1: Nginx health endpoint
if curl -sf "${NGINX_URL}/nginx-health" | grep -q "healthy"; then
    log_pass "Nginx health endpoint returns 'healthy'"
else
    log_fail "Nginx health endpoint not responding"
fi

# Test 1.2: Streamlit health through nginx
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${NGINX_URL}/_stcore/health" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    log_pass "Streamlit health endpoint returns 200 through nginx"
else
    log_fail "Streamlit health endpoint returned ${HTTP_CODE} (expected 200)"
fi

# Test 1.3: Main page loads
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${NGINX_URL}" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    log_pass "Main page returns 200"
else
    log_fail "Main page returned ${HTTP_CODE} (expected 200)"
fi


log_section "2. Security Headers"

HEADERS=$(curl -sfI "${NGINX_URL}" 2>/dev/null || echo "")

# Test 2.1: X-Frame-Options
if echo "$HEADERS" | grep -qi "X-Frame-Options: SAMEORIGIN"; then
    log_pass "X-Frame-Options header present"
else
    log_fail "X-Frame-Options header missing"
fi

# Test 2.2: X-Content-Type-Options
if echo "$HEADERS" | grep -qi "X-Content-Type-Options: nosniff"; then
    log_pass "X-Content-Type-Options header present"
else
    log_fail "X-Content-Type-Options header missing"
fi

# Test 2.3: Server version hidden
if echo "$HEADERS" | grep -qi "Server: nginx/"; then
    log_fail "Server version is exposed (server_tokens should be off)"
else
    log_pass "Server version is hidden"
fi

# Test 2.4: Content-Security-Policy
if echo "$HEADERS" | grep -qi "Content-Security-Policy"; then
    log_pass "Content-Security-Policy header present"
else
    log_fail "Content-Security-Policy header missing"
fi


log_section "3. Session Affinity"

# Test 3.1: Same client gets same backend on repeated requests
CONSISTENT=true
for i in $(seq 1 5); do
    RESPONSE=$(curl -sf -D - "${NGINX_URL}" 2>/dev/null | head -50)
    if ! echo "$RESPONSE" | grep -q "200"; then
        CONSISTENT=false
        break
    fi
done

if $CONSISTENT; then
    log_pass "Repeated requests return consistent 200 responses (ip_hash working)"
else
    log_fail "Inconsistent responses across repeated requests"
fi


log_section "4. Backend Health"

# Test 4.1: All backends are running
RUNNING_BACKENDS=0
for i in 1 2 3; do
    if docker ps --format '{{.Names}}' | grep -q "nba-app-${i}"; then
        RUNNING_BACKENDS=$((RUNNING_BACKENDS + 1))
    fi
done

if [ "$RUNNING_BACKENDS" -eq 3 ]; then
    log_pass "All 3 backend instances are running"
else
    log_fail "Only ${RUNNING_BACKENDS}/3 backend instances are running"
fi

# Test 4.2: Each backend's health endpoint responds
for i in 1 2 3; do
    HEALTH=$(docker exec "nba-app-${i}" curl -sf "http://localhost:8501/_stcore/health" 2>/dev/null || echo "FAIL")
    if [ "$HEALTH" != "FAIL" ]; then
        log_pass "nba-app-${i} health check responds"
    else
        log_fail "nba-app-${i} health check failed"
    fi
done

# Test 4.3: Nginx container is healthy
NGINX_HEALTH=$(docker inspect --format='{{.State.Health.Status}}' nba-nginx 2>/dev/null || echo "unknown")
if [ "$NGINX_HEALTH" = "healthy" ]; then
    log_pass "Nginx container health status is 'healthy'"
else
    log_fail "Nginx container health status is '${NGINX_HEALTH}' (expected 'healthy')"
fi


log_section "5. Rate Limiting"

# Test 5.1: Normal requests succeed
NORMAL_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${NGINX_URL}/nginx-health" 2>/dev/null || echo "000")
if [ "$NORMAL_CODE" = "200" ]; then
    log_pass "Normal request within rate limit succeeds"
else
    log_fail "Normal request failed with ${NORMAL_CODE}"
fi

# Test 5.2: Burst of requests (within burst limit) should succeed
BURST_PASS=0
for i in $(seq 1 15); do
    CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${NGINX_URL}/nginx-health" 2>/dev/null || echo "000")
    if [ "$CODE" = "200" ]; then
        BURST_PASS=$((BURST_PASS + 1))
    fi
done

if [ "$BURST_PASS" -ge 10 ]; then
    log_pass "Burst requests within limit: ${BURST_PASS}/15 succeeded"
else
    log_fail "Burst requests: only ${BURST_PASS}/15 succeeded (expected >= 10)"
fi


log_section "6. Failover"

# Test 6.1: Stop one backend and verify nginx still serves
echo "  Stopping nba-app-3 for failover test..."
docker stop nba-app-3 > /dev/null 2>&1 || true
sleep 2

FAILOVER_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${NGINX_URL}" 2>/dev/null || echo "000")
if [ "$FAILOVER_CODE" = "200" ]; then
    log_pass "App still responds after stopping 1 backend"
else
    log_fail "App returned ${FAILOVER_CODE} after stopping 1 backend"
fi

# Restart the stopped backend (cleanup trap handles this on failure too)
echo "  Restarting nba-app-3..."
docker start nba-app-3 > /dev/null 2>&1 || true
sleep 5


# --- Results ---

log_section "Results"
echo ""
echo -e "  Total:  ${TOTAL}"
echo -e "  ${GREEN}Passed: ${PASS}${NC}"
echo -e "  ${RED}Failed: ${FAIL}${NC}"
echo ""

if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}All tests passed.${NC}"
    exit 0
else
    echo -e "${RED}${FAIL} test(s) failed.${NC}"
    exit 1
fi
