# ─────────────────────────────────────────────────────────────────────────────
# FraudGuard Frontend — Dockerfile
# Optimised for HuggingFace Spaces (Docker SDK, port 7860).
#
# Build stages:
#   builder  — Node 22 Alpine: installs deps and runs `vite build`
#   runtime  — Nginx Alpine:   serves the compiled static files
#
# Node 22 is required because:
#   • vite@8          requires node ^20.19.0 || >=22.12.0
#   • @vitejs/plugin-react@6 requires node ^20.19.0 || >=22.12.0
#   • rolldown (vite internals) requires node ^20.19.0 || >=22.12.0
#   Node 18 is EOL for these packages and will fail npm ci.
#
# Why Nginx instead of HF Static SDK?
#   The app is a React SPA. Any URL that isn't "/" must return index.html
#   so the client-side router can handle it. Static SDK serves files
#   verbatim and has no fallback route — Nginx's try_files solves this.
#
# Why port 7860?
#   HuggingFace Spaces Docker SDK REQUIRES the container to listen on 7860.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Node build ───────────────────────────────────────────────────────
FROM node:22-alpine AS builder

WORKDIR /app

# Copy manifests first so Docker caches the npm install layer separately.
# This layer is only invalidated when package.json or package-lock.json change,
# not on every source code edit — keeps rebuilds fast.
COPY package.json package-lock.json ./

# npm install (not ci) to handle any lockfile drift gracefully,
# while still installing exact versions where recorded.
# --prefer-offline uses the cache when available; --no-audit speeds up CI.
RUN npm install --prefer-offline --no-audit --no-fund

# Copy source files and build
COPY . .

RUN npm run build
# Produces /app/dist/


# ── Stage 2: Nginx runtime ────────────────────────────────────────────────────
FROM nginx:1.27-alpine AS runtime

# Replace Nginx default config with our SPA-aware config
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/app.conf

# Copy the compiled React app from the builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# ── Security: run Nginx as non-root (UID 1000) ───────────────────────────────
# HuggingFace Spaces runs containers as UID 1000 by default.
# Grant that UID ownership of all directories Nginx needs to write at runtime.
RUN chown -R 1000:1000 \
        /usr/share/nginx/html \
        /var/cache/nginx \
        /var/log/nginx \
        /etc/nginx/conf.d \
    && touch /var/run/nginx.pid \
    && chown 1000:1000 /var/run/nginx.pid

USER 1000

# ── HuggingFace Spaces requires port 7860 ────────────────────────────────────
EXPOSE 7860

# ── Health check ─────────────────────────────────────────────────────────────
# wget is available in Alpine's Nginx image; curl is not installed by default.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost:7860/health || exit 1

CMD ["nginx", "-g", "daemon off;"]
