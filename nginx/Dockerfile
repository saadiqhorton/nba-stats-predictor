FROM nginx:1.25-alpine

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom load balancer configuration
COPY nginx.conf /etc/nginx/conf.d/nginx.conf

# Expose HTTP port
EXPOSE 80

# Health check: verify nginx is responding
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO- http://localhost/nginx-health || exit 1
