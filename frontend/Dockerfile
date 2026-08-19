# Multi-stage production container for Vite + React 19 Frontend
# Stage 1: Build static distribution bundle
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

# Public build-time API base URL (injected during Railway build)
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

COPY . ./
RUN npm run build

# Stage 2: Serve static assets with Caddy
FROM caddy:2-alpine

COPY --from=builder /app/dist /srv
COPY Caddyfile /etc/caddy/Caddyfile

EXPOSE 80

CMD ["caddy", "run", "--config", "/etc/caddy/Caddyfile", "--adapter", "caddyfile"]
