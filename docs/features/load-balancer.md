# Load Balancer

## What This Does

The load balancer spreads incoming user traffic across multiple copies of the app. Instead of one app instance handling all users, Nginx sits in front and forwards each request to one of three backend instances.

## Why It Matters

A single Streamlit instance can handle about 20-30 users at the same time. With the load balancer, the app supports 60-90 concurrent users. If one instance crashes, the other two keep running. Users connected to healthy instances are not affected.

## How to Use It

### Starting the Load-Balanced App

1. Make sure Docker and Docker Compose are installed.

2. Build and start all containers:
   ```bash
   docker compose up -d --build
   ```

3. Open `http://localhost:8088` in your browser.

4. Verify all containers are healthy:
   ```bash
   docker compose ps
   ```
   All four containers should show `healthy` in the status column.

### Scaling the Number of Instances

The default setup runs 3 app instances. You can change this from 1 to 8.

1. Run the scale script with the number you want:
   ```bash
   ./scripts/scale.sh 5
   ```

2. Rebuild and restart:
   ```bash
   docker compose up -d --build
   ```

### Reloading Nginx Configuration

If you change `nginx/nginx.conf`, you can reload it without stopping the app:

```bash
./scripts/reload_nginx.sh
```

This checks the configuration for errors first. If the config is valid, Nginx reloads without dropping active connections.

### Stopping Everything

```bash
docker compose down
```

## How It Works

### Request Routing

When a browser connects to `http://localhost:8088`, the request goes to Nginx. Nginx picks which app instance handles the request based on the user's IP address. All future requests from that same IP go to the same instance.

This is called IP-based session routing. It is needed because each Streamlit instance stores session data in its own memory. If a user's requests went to different instances, their session data would be lost.

### Health Monitoring

Each app instance has a health check that runs every 30 seconds. It sends a request to Streamlit's built-in health endpoint (`/_stcore/health`). If an instance fails 3 checks in a row, Nginx stops sending it traffic for 30 seconds. After 30 seconds, Nginx tries again.

Nginx also has its own health check at `/nginx-health`. Docker uses this to know if Nginx itself is running.

### WebSocket Support

Streamlit uses WebSocket connections for real-time updates between the browser and the server. Nginx is configured to pass WebSocket upgrade requests through to the backend instances. The connection stays open for up to 24 hours.

### Rate Limiting

Each user IP address is limited to 10 requests per second. Short bursts of up to 20 requests are allowed. If a user exceeds this limit, they get a 429 (Too Many Requests) response. Health check endpoints are excluded from rate limiting.

## Architecture

```
Browser (port 8088)
    |
    v
Nginx (port 80 inside Docker)
    |
    |-- IP hash routing
    |
    +-------+-------+
    |       |       |
    v       v       v
 App 1   App 2   App 3
(:8501)  (:8501)  (:8501)
```

All containers run on a shared Docker network called `nba-network`. The app instances are only reachable from inside this network. Only Nginx's port (mapped to 8088 on the host) is accessible from outside.

## Configuration Options

### Nginx Settings (nginx/nginx.conf)

| Setting | Default | What It Controls |
|---------|---------|-----------------|
| `rate` in `limit_req_zone` | `10r/s` | Maximum requests per second per user |
| `burst` in `limit_req` | `20` | Extra requests allowed in a short burst |
| `max_fails` | `3` | Failed requests before marking a backend as down |
| `fail_timeout` | `30s` | How long a failed backend stays out of rotation |
| `proxy_read_timeout` | `86400s` | Maximum time a WebSocket connection can stay open (24 hours) |
| `proxy_connect_timeout` | `5s` | How long Nginx waits to connect to a backend |

### Docker Compose Settings (docker-compose.yml)

| Setting | Default | What It Controls |
|---------|---------|-----------------|
| Nginx port mapping | `8088:80` | The port users connect to on the host machine |
| Health check interval | `30s` | How often each container's health is checked |
| Health check retries | `3` | Failed checks before a container is marked unhealthy |
| Start period | `40s` | Grace period for the app to start before health checks begin |

### Scale Script Settings (scripts/scale.sh)

| Setting | Default | What It Controls |
|---------|---------|-----------------|
| Minimum instances | `1` | Fewest instances allowed |
| Maximum instances | `8` | Most instances allowed |

## Common Issues

### App shows "502 Bad Gateway"
**What you see:** A blank page or a "502 Bad Gateway" error.
**Why it happens:** None of the backend instances are ready yet, or all of them have crashed.
**How to fix it:** Wait 40-60 seconds for the instances to start. If the error persists, check container logs:
```bash
docker compose logs nba-app-1
```

### Session data lost after restart
**What you see:** Your search results disappear and you have to start over.
**Why it happens:** Each app instance stores session data in memory. When an instance restarts, that memory is cleared.
**How to fix it:** Refresh the page. Nginx will route you to a healthy instance.

### Scale script fails with "not a valid number"
**What you see:** An error message from the scale script.
**Why it happens:** The script only accepts a whole number between 1 and 8.
**How to fix it:** Run the script with a valid number:
```bash
./scripts/scale.sh 3
```

### Rate limit errors (429 responses)
**What you see:** Some requests fail with a 429 status code.
**Why it happens:** You exceeded 10 requests per second from your IP address.
**How to fix it:** Wait a moment and try again. The limit resets quickly.

### Uneven traffic distribution
**What you see:** One instance is handling more users than others.
**Why it happens:** IP-based routing can be uneven when many users share the same IP address (for example, users behind a company network).
**How to fix it:** This is a known limitation. For most use cases, the distribution is adequate. If it becomes a problem, consider switching to cookie-based session routing.

## Related Features

- [Prediction Engine](./prediction-engine.md) - The core app that runs on each backend instance
